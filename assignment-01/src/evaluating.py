import torch
import torch.nn.functional as F
from skimage.segmentation import watershed
import numpy as np
from scipy import ndimage

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')
IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)


# ---------------------------------------------------------------------------
# 2. Extração de instâncias ingênua: limiar + componentes conexos
# ---------------------------------------------------------------------------
def extract_instances(prob_map, threshold=0.5, min_size=5):
    """
    prob_map: array HxW de probabilidades (saída do sigmoid).
    Retorna lista de máscaras binárias HxW, uma por componente conexa
    (conectividade 8, via scipy.ndimage.label), descartando ruído < min_size px.
    """
    binary = (prob_map > threshold).astype(np.uint8)
    structure = np.ones((3, 3), dtype=np.uint8)  # conectividade-8
    labeled, num = ndimage.label(binary, structure=structure)
    instances = []
    for i in range(1, num + 1):
        inst_mask = (labeled == i).astype(np.uint8)
        if inst_mask.sum() >= min_size:
            instances.append(inst_mask)
    return instances


# ---------------------------------------------------------------------------
# 3 e 4. Avaliação como instâncias — matching guloso por IoU decrescente
# ---------------------------------------------------------------------------
#
# REGRA DE MATCHING (documentada, escolha do grupo): guloso por IoU decrescente.
#   1. Calcula a matriz de IoU (pred x gt) entre todas as máscaras preditas
#      e todas as máscaras verdadeiras da imagem.
#   2. Lista todos os pares (pred, gt) com seu IoU e ordena por IoU decrescente.
#   3. Percorre a lista em ordem; casa o par se IoU >= limiar do momento E se
#      nem a predição nem o GT já foram usados em outro casamento.
#   4. Predições não casadas => FP. GTs não casados => FN. Pares casados => TP.
#      Cada instância prevista casa com NO MÁXIMO uma verdadeira, e vice-versa.
#
# (Alternativa seria o algoritmo Hungarian/linear_sum_assignment, que resolve
#  o casamento ótimo global em vez de guloso; optamos pelo guloso por ser o
#  padrão adotado na métrica oficial do próprio DSB2018 e por não exigir
#  matriz de custo quadrada/completa.)

# AP por limiar: como o método ingênuo (limiar + componentes conexos) não
# produz um score de confiança por instância, não há curva precisão-recall
# para integrar. Seguimos a métrica oficial do DSB2018: em cada limiar de
# IoU, AP = TP / (TP + FP + FN), agregado sobre TODAS as imagens do split.
# mAP = média de AP sobre os limiares 0.50:0.95:0.05.


@torch.no_grad()
def evaluate_instances_binary(model, loader, prob_threshold=0.5, min_size=5, k_samples=7):
    model.eval()

    n_threshold = len(IOU_THRESHOLDS)
    tp_total = np.zeros(n_threshold)
    denom_total = np.zeros(n_threshold)

    count_errors = []
    densities = []
    per_image_aps = []
    samples = []

    for images, _, instance_masks_batch, _ in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        for b in range(images.size(0)):
            pred_masks = extract_instances(probs[b, 0], prob_threshold, min_size)
            gt_masks = instance_masks_batch[b]
            n_pred, n_gt = len(pred_masks), len(gt_masks)
            count_error = abs(n_pred - n_gt)
            count_errors.append(count_error)
            densities.append(n_gt)

            img_plot = images[b].cpu().numpy().transpose(1, 2, 0)
            samples.append((count_error, img_plot, gt_masks, pred_masks))

            if n_pred == 0 and n_gt == 0:
                per_image_aps.append(1.0)
                continue 

            if n_pred == 0 or n_gt == 0:
                denom_total += n_pred + n_gt
                per_image_aps.append(0.0)
                continue

            pairs = []
            img_aps = np.zeros(n_threshold)


            pred_flat = np.array(pred_masks, dtype=np.int32).reshape(n_pred, -1)
            gt_flat = np.array(gt_masks, dtype=np.int32).reshape(n_gt, -1)
            inter_matrix = np.dot(pred_flat, gt_flat.T)
            area_pred = pred_flat.sum(axis=1)[:, np.newaxis]
            area_gt = gt_flat.sum(axis=1)[np.newaxis, :]
            union_matrix = area_pred + area_gt - inter_matrix
            iou_matrix = inter_matrix / np.maximum(union_matrix, 1e-6)
            pairs = [(iou_matrix[i, j], i, j) 
                    for i in range(n_pred) for j in range(n_gt) 
                    if iou_matrix[i, j] > 0
            ]
            pairs.sort(key=lambda x: -x[0])

            for k, t in enumerate(IOU_THRESHOLDS):
                matched_pred, matched_gt = set(), set()
                tp = 0
                for iou, i, j in pairs:
                    if iou < t:
                        break
                    if i not in matched_pred and j not in matched_gt:
                        matched_pred.add(i)
                        matched_gt.add(j)
                        tp += 1
                fp = n_pred - tp
                fn = n_gt - tp
                denom = tp + fp + fn
                img_aps[k] = tp / denom if denom > 0 else 1.0

                tp_total[k] += tp
                denom_total[k] += denom

            per_image_aps.append(float(img_aps.mean()))

    aps = np.where(denom_total > 0, tp_total / denom_total, 1.0)
    samples.sort(key=lambda s: -s[0])
    return aps, count_errors, densities, per_image_aps, samples[:k_samples]



# ==============================================================================
# 5. DECODIFICAÇÃO POR WATERSHED MARCADO E AVALIAÇÃO DE INSTÂNCIAS
# ==============================================================================
def decode_watershed(
		class_probs,
		dist_map=None,
		interior_thresh=0.4,
		fg_thresh=0.5,
		min_marker_size=5,
):
	"""Decodificação por Watershed Marcado:

	1. Marcadores: Componentes conexos do Interior binarizado.
	2. Máscara de Foreground: Regiões de (Interior + Fronteira).
	3. Relevo de Inundação: Inverso da distância contínua (-dist_map) ou do
	interior (-prob_interior).
	"""
	p_bg, p_interior, p_boundary = class_probs

	# 1. Marcadores das sementes
	marker_bin = p_interior > interior_thresh
	labeled_markers, num_markers = ndimage.label(
			marker_bin, structure=np.ones((3, 3))
	)

	for i in range(1, num_markers + 1):
		if (labeled_markers == i).sum() < min_marker_size:
			labeled_markers[labeled_markers == i] = 0

	if labeled_markers.max() == 0:
		return []
		
	# 2. Máscara que impede invasão do fundo
	fg_mask = (p_interior + p_boundary) > fg_thresh

	# 3. Superfície topográfica invertida (centros são vales, bordas são cristas)
	elevation = -dist_map if dist_map is not None else -p_interior

	ws_labels = watershed(elevation, markers=labeled_markers, mask=fg_mask)

	pred_masks = []
	for i in range(1, ws_labels.max() + 1):
		inst = (ws_labels == i).astype(np.uint8)
		if inst.sum() >= min_marker_size:
			pred_masks.append(inst)

	return pred_masks


@torch.no_grad()
def evaluate_instances_ternary(
		model,
		loader,
		interior_thresh=0.55,
		fg_thresh=0.5,
		min_marker_size=5,
		k_samples=6,
):
	"""Avaliação idêntica à Parte 1: matching guloso por IoU decrescente."""
	model.eval()
	n_threshold = len(IOU_THRESHOLDS)
	tp_total = np.zeros(n_threshold)
	denom_total = np.zeros(n_threshold)

	count_errors = []
	densities = []
	per_image_aps = []
	samples = []

	for images, _, _, instance_masks_batch, _ in loader:
		images = images.to(DEVICE)
		logits_cls, dist_pred = model(images)
		probs_batch = F.softmax(logits_cls, dim=1).cpu().numpy()
		dists_batch = dist_pred.squeeze(1).cpu().numpy()

		for b in range(images.size(0)):
			pred_masks = decode_watershed(
					probs_batch[b],
					dists_batch[b],
					interior_thresh=interior_thresh,
					fg_thresh=fg_thresh,
					min_marker_size=min_marker_size,
			)
			gt_masks = instance_masks_batch[b]
			n_pred, n_gt = len(pred_masks), len(gt_masks)

			count_err = abs(n_pred - n_gt)
			count_errors.append(count_err)
			densities.append(n_gt)

			img_np = images[b].cpu().numpy().transpose(1, 2, 0)
			samples.append((count_err, img_np, gt_masks, pred_masks))

			if n_pred == 0 and n_gt == 0:
				per_image_aps.append(1.0)
				continue
			if n_pred == 0 or n_gt == 0:
				denom_total += n_pred + n_gt
				per_image_aps.append(0.0)
				continue

			# Matriz de IoU Vetorizada
			pred_flat = np.array(pred_masks, dtype=np.int32).reshape(n_pred, -1)
			gt_flat = np.array(gt_masks, dtype=np.int32).reshape(n_gt, -1)
			inter = np.dot(pred_flat, gt_flat.T)
			area_pred = pred_flat.sum(axis=1)[:, np.newaxis]
			area_gt = gt_flat.sum(axis=1)[np.newaxis, :]
			union = area_pred + area_gt - inter
			iou_matrix = inter / np.maximum(union, 1e-6)

			pairs = [
					(iou_matrix[i, j], i, j)
					for i in range(n_pred)
					for j in range(n_gt)
					if iou_matrix[i, j] > 0
			]
			pairs.sort(key=lambda x: -x[0])

			img_aps = np.zeros(n_threshold)
			for k, t in enumerate(IOU_THRESHOLDS):
				matched_pred, matched_gt = set(), set()
				tp = 0
				for iou, i, j in pairs:
					if iou < t:
						break
					if i not in matched_pred and j not in matched_gt:
						matched_pred.add(i)
						matched_gt.add(j)
						tp += 1
				fp = n_pred - tp
				fn = n_gt - tp
				denom = tp + fp + fn
				img_aps[k] = tp / denom if denom > 0 else 1.0

				tp_total[k] += tp
				denom_total[k] += denom

			per_image_aps.append(float(img_aps.mean()))

	aps = np.where(denom_total > 0, tp_total / denom_total, 1.0)
	samples.sort(key=lambda s: -s[0])
	return aps, count_errors, densities, per_image_aps, samples[:k_samples]

