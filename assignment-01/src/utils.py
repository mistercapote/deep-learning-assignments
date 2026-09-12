import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from evaluating import decode_watershed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')


def plot_metrics(all_mAPs, all_count_errors, all_densities):
    print(f"Média global mAP: {np.mean(all_mAPs):.4f}")
    print(f"Média global Erro Absoluto de Contagem: {np.mean(all_count_errors):.4f}")

    corr_ap = np.corrcoef(all_densities, all_mAPs)[0, 1]
    print(f'Correlação (densidade x mAP por imagem): {corr_ap:.3f}')

    corr_err = np.corrcoef(all_densities, all_count_errors)[0, 1]
    print(f'Correlação (densidade x erro de contagem): {corr_err:.3f}')

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    color = 'tab:blue'
    ax[0].set_xlabel('Densidade de Objetos (Qtd. de Instâncias Reais)')
    ax[0].set_ylabel('mAP (Threshold 0.5 a 0.95)', color=color)
    ax[0].scatter(all_densities, all_mAPs, color=color, alpha=0.6, label='mAP')
    ax[0].tick_params(axis='y', labelcolor=color)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_title("Quantificação de Falhas: Desempenho vs. Densidade", fontsize=10, fontweight='bold')

    color = 'tab:red'
    ax[1].set_ylabel('Erro Absoluto de Contagem', color=color)
    ax[1].scatter(all_densities, all_count_errors, color=color, alpha=0.6, label='Erro de Contagem')
    ax[1].set_xlabel('Densidade de Objetos (Qtd. de Instâncias Reais)')
    ax[1].tick_params(axis='y', labelcolor=color)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_title("Quantificação de Falhas: Desempenho vs. Densidade", fontsize=10, fontweight='bold')
    plt.grid(False)
    fig.tight_layout()
    plt.show()

    
def masks_to_label(masks, shape):
    """
    Converte uma LISTA de máscaras binárias (formato usado por
    evaluate_instances/greedy_match) num único MAPA DE RÓTULOS (H,W),
    formato que cmap='nipy_spectral' espera pra colorir por instância:
    0 = fundo, 1 = instância 1, 2 = instância 2, ...
 
    Se duas máscaras se sobrepuserem (não deveria acontecer com
    componentes conexos, mas por segurança), a última da lista "ganha"
    o pixel disputado.
    """
    label = np.zeros(shape, dtype=np.int32)
    for i, m in enumerate(masks, start=1):
        label[m.astype(bool)] = i
    return label

    
def plot_samples(samples):
    """
    Layout: uma coluna por amostra, 3 linhas (original / gabarito / predição),
    cada instância com uma cor distinta via nipy_spectral (fundo forçado a
    preto com máscara, pra não ficar colorido também).
    """
    n_samples = len(samples)
    fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 12), squeeze=False)
    fig.suptitle(f"Os {n_samples} piores resultados", fontsize=14, fontweight='bold')
 
    cmap = plt.get_cmap('prism').copy()
    cmap.set_bad('black')  # fundo (rótulo 0, mascarado) sempre preto
 
    for idx, (err, img, gt_masks, pred_masks) in enumerate(samples):
        H, W = img.shape[:2]
        gt_label = masks_to_label(gt_masks, (H, W))
        pred_label = masks_to_label(pred_masks, (H, W))
 
        gt_masked = np.ma.masked_where(gt_label == 0, gt_label)
        pred_masked = np.ma.masked_where(pred_label == 0, pred_label)

        axes[0, idx].imshow(img)
        axes[0, idx].set_title("Original")
        axes[0, idx].axis('off')

        axes[1, idx].imshow(gt_masked, cmap=cmap, interpolation='nearest')
        axes[1, idx].set_title(f"Gabarito (Instâncias: {len(gt_masks)})")
        axes[1, idx].axis('off')
 
        axes[2, idx].imshow(pred_masked, cmap=cmap, interpolation='nearest')
        axes[2, idx].set_title(f"Predição (Instâncias: {len(pred_masks)}) | Erro: {err}")
        axes[2, idx].axis('off')
 
    plt.tight_layout()
    plt.show()


from matplotlib.colors import ListedColormap


def plot_ternary_masks(model, loader, n_samples=4):
	"""NOVA FUNÇÃO SOLICITADA:

	Visualiza detalhadamente as predições intermediárias ternárias e contínuas:
	- Linha 1: Imagem Original
	- Linha 2: Mapa de Distância Contínuo (Predito)
	- Linha 3: Rótulo Ternário Gabarito (GT)
	- Linha 4: Rótulo Ternário Predito (Argmax da CNN)
	- Linha 5: Instâncias Finais Segmentadas pelo Watershed
	"""
	model.eval()
	cmap_ternary = ListedColormap(['#1a1a1a', '#2b83ba', '#d7191c'])
	legend_elements = [
			mpatches.Patch(color='#1a1a1a', label='0: Fundo'),
			mpatches.Patch(color='#2b83ba', label='1: Interior'),
			mpatches.Patch(color='#d7191c', label='2: Fronteira'),
	]

	images, labels, dists, instances, ids = next(iter(loader))
	images = images.to(DEVICE)

	with torch.no_grad():
		logits_cls, dist_pred = model(images)
		probs = F.softmax(logits_cls, dim=1).cpu().numpy()
		pred_ternary = np.argmax(probs, axis=1)
		dist_pred_np = dist_pred.squeeze(1).cpu().numpy()

	fig, axes = plt.subplots(
			5, n_samples, figsize=(4 * n_samples, 16), squeeze=False
	)
	fig.suptitle(
			'Análise da Representação Ternária e Watershed (Parte 2)',
			fontsize=14,
			fontweight='bold',
	)

	cmap_inst = plt.get_cmap('prism').copy()
	cmap_inst.set_bad('black')

	for idx in range(min(n_samples, images.size(0))):
		img_show = images[idx].cpu().numpy().transpose(1, 2, 0)
		gt_ternary = labels[idx].numpy()
		pred_t = pred_ternary[idx]
		pred_d = dist_pred_np[idx]

		# Decodifica instâncias com watershed para visualização
		instances_pred = decode_watershed(probs[idx], pred_d)
		label_ws = np.zeros(img_show.shape[:2], dtype=np.int32)
		for i, m in enumerate(instances_pred, start=1):
			label_ws[m.astype(bool)] = i
		ws_masked = np.ma.masked_where(label_ws == 0, label_ws)

		# 1. Original
		axes[0, idx].imshow(img_show)
		axes[0, idx].set_title(f'Original: {ids[idx][:8]}')
		axes[0, idx].axis('off')

		# 2. Distância Predita
		im_d = axes[1, idx].imshow(pred_d, cmap='viridis', vmin=0, vmax=1)
		axes[1, idx].set_title('Distância ao Fundo (Predita)')
		axes[1, idx].axis('off')

		# 3. Gabarito Ternário
		axes[2, idx].imshow(
				gt_ternary, cmap=cmap_ternary, vmin=0, vmax=2, interpolation='nearest'
		)
		axes[2, idx].set_title('Gabarito Ternário (GT)')
		axes[2, idx].axis('off')

		# 4. Predição Ternária
		axes[3, idx].imshow(
				pred_t, cmap=cmap_ternary, vmin=0, vmax=2, interpolation='nearest'
		)
		axes[3, idx].set_title('Predição Ternária (CNN)')
		axes[3, idx].axis('off')

		# 5. Instâncias Watershed
		axes[4, idx].imshow(ws_masked, cmap=cmap_inst, interpolation='nearest')
		axes[4, idx].set_title(f'Instâncias Watershed ({len(instances_pred)})')
		axes[4, idx].axis('off')

	fig.legend(
			handles=legend_elements,
			loc='lower center',
			ncol=3,
			bbox_to_anchor=(0.5, 0.01),
			frameon=True,
	)
	plt.tight_layout(rect=[0, 0.03, 1, 0.98])
	plt.show()



def plot_training_curves_binary(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'])
    axes[0].set_title('Loss de treino (BCE + Dice)')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['val_iou'], label='IoU')
    axes[1].plot(history['val_dice'], label='Dice')
    axes[1].set_title('Métricas de validação (nível de pixel)')
    axes[1].set_xlabel('Época')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()



def plot_training_curves_ternary(history):
	fig, axes = plt.subplots(1, 2, figsize=(12, 4))
	axes[0].plot(history['train_loss'], label='Treino (Total)')
	axes[0].plot(history['val_loss'], label='Validação (Total)')
	axes[0].set_title('Evolução da Loss Combinada')
	axes[0].set_xlabel('Época')
	axes[0].set_ylabel('Loss')
	axes[0].grid(alpha=0.3)
	axes[0].legend()

	axes[1].plot(history['val_cls_loss'], label='Val Classif. (Focal)')
	axes[1].plot(history['val_dist_loss'], label='Val Distância (L1)')
	axes[1].set_title('Componentes da Loss de Validação')
	axes[1].set_xlabel('Época')
	axes[1].grid(alpha=0.3)
	axes[1].legend()

	plt.tight_layout()
	plt.show()



def plot_metrics(all_mAPs, all_count_errors, all_densities):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    color = 'tab:blue'
    ax[0].set_xlabel('Densidade de Objetos (Qtd. de Instâncias Reais)')
    ax[0].set_ylabel('mAP (Threshold 0.5 a 0.95)', color=color)
    ax[0].scatter(all_densities, all_mAPs, color=color, alpha=0.6, label='mAP')
    ax[0].tick_params(axis='y', labelcolor=color)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_title("Quantificação de Falhas: Desempenho vs. Densidade", fontsize=10, fontweight='bold')

    color = 'tab:red'
    ax[1].set_ylabel('Erro Absoluto de Contagem', color=color)
    ax[1].scatter(all_densities, all_count_errors, color=color, alpha=0.6, label='Erro de Contagem')
    ax[1].set_xlabel('Densidade de Objetos (Qtd. de Instâncias Reais)')
    ax[1].tick_params(axis='y', labelcolor=color)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_title("Quantificação de Falhas: Desempenho vs. Densidade", fontsize=10, fontweight='bold')
    plt.grid(False)
    fig.tight_layout()
    plt.show()


def masks_to_label(masks, shape):
    """
    Converte uma LISTA de máscaras binárias (formato usado por
    evaluate_instances/greedy_match) num único MAPA DE RÓTULOS (H,W),
    formato que cmap='nipy_spectral' espera pra colorir por instância:
    0 = fundo, 1 = instância 1, 2 = instância 2, ...
 
    Se duas máscaras se sobrepuserem (não deveria acontecer com
    componentes conexos, mas por segurança), a última da lista "ganha"
    o pixel disputado.
    """
    label = np.zeros(shape, dtype=np.int32)
    for i, m in enumerate(masks, start=1):
        label[m.astype(bool)] = i
    return label

    
def plot_samples(samples, part: int = 1):
    """
    Layout: uma coluna por amostra, 3 linhas (original / gabarito / predição),
    cada instância com uma cor distinta via nipy_spectral (fundo forçado a
    preto com máscara, pra não ficar colorido também).
    """
    n_samples = len(samples)
    fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 12), squeeze=False)
    fig.suptitle(f"Amostra de {n_samples} Resultados - Parte {part}", fontsize=14, fontweight='bold')
 
    cmap = plt.get_cmap('prism').copy()
    cmap.set_bad('black')  # fundo (rótulo 0, mascarado) sempre preto
 
    for idx, (err, img, gt_masks, pred_masks) in enumerate(samples):
        H, W = img.shape[:2]
        gt_label = masks_to_label(gt_masks, (H, W))
        pred_label = masks_to_label(pred_masks, (H, W))
 
        gt_masked = np.ma.masked_where(gt_label == 0, gt_label)
        pred_masked = np.ma.masked_where(pred_label == 0, pred_label)

        axes[0, idx].imshow(img)
        axes[0, idx].set_title("Original")
        axes[0, idx].axis('off')

        axes[1, idx].imshow(gt_masked, cmap=cmap, interpolation='nearest')
        axes[1, idx].set_title(f"Gabarito (Instâncias: {len(gt_masks)})")
        axes[1, idx].axis('off')
 
        axes[2, idx].imshow(pred_masked, cmap=cmap, interpolation='nearest')
        axes[2, idx].set_title(f"Predição (Instâncias: {len(pred_masks)}) | Erro: {err}")
        axes[2, idx].axis('off')
 
    plt.tight_layout()
    plt.show()
