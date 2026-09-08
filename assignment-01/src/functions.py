import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from typing import Dict, List, Sequence, Tuple
from skimage.segmentation import watershed
import scipy.ndimage as ndi
from torch.nn import functional as F
from skimage.feature import peak_local_max


def training1(model: nn.Module, dataloader: DataLoader, device: torch.device, lr: int = 1e-4, num_epochs: int = 10) -> dict:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_batches = len(dataloader)

    print(f"Treinando em: {device}")
    print(f"Batches por época: {num_batches}")

    history = {'loss': [], 'iou': [], 'dice': []}
    for epoch in range(num_epochs):
        model.train()

        accumulated_loss = torch.tensor(0.0, device=device)
        total_intersection = torch.tensor(0.0, device=device)
        total_union = torch.tensor(0.0, device=device)

        for images, masks, _ in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()

            prediction = model(images)
            loss = criterion(prediction, masks)
            
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_bin = (prediction > 0).float()
                intersection = (pred_bin * masks).sum()
                union = pred_bin.sum() + masks.sum() - intersection
                
                accumulated_loss += loss.detach()
                total_intersection += intersection
                total_union += union

        epoch_loss = accumulated_loss.item() / num_batches
        ti = total_intersection.item()
        tu = total_union.item()

        epoch_iou = ti / (tu + 1e-6)
        epoch_dice = (2.0 * ti) / (tu + ti + 1e-6)

        history['loss'].append(epoch_loss)
        history['iou'].append(epoch_iou)
        history['dice'].append(epoch_dice)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}")

    return history


def dice_loss(probs, target, class_idx, eps=1e-6):
    p = probs[:, class_idx]
    t = (target == class_idx).float()
    inter = (p * t).sum()
    return 1 - (2*inter + eps) / (p.sum() + t.sum() + eps)


def training2(
    model: nn.Module,
    dataset,
    dataloader: DataLoader,
    device: torch.device,
    lr: float = 1e-4,
    num_epochs: int = 10,
    dice_weight_interior: float = 1.5,   # antes: peso implícito de 1.0
    dice_weight_boundary: float = 1.0,   # antes: 2.0 (baseline p/ comparar)
) -> dict:

    class_counts = torch.zeros(3)
    for _, mask, _, _ in dataset:               # AGORA 4 itens
        for c in range(3):
            class_counts[c] += (mask == c).sum()

    total = class_counts.sum()
    weights = total / (3 * class_counts)
    weights = weights.clamp(max=weights.median() * 10)
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')   # MUDOU: reduction none
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_batches = len(dataloader)

    print(f"Treinando em: {device}")
    print(f"Batches por época: {num_batches}")
    print(f"Pesos CE (bg/interior/fronteira): {weights.tolist()}")
    print(f"Pesos Dice -> interior: {dice_weight_interior} | fronteira: {dice_weight_boundary}")

    history = {'loss': [], 'iou': [], 'dice': []}
    for epoch in range(num_epochs):
        model.train()

        accumulated_loss = torch.tensor(0.0, device=device)
        total_intersection = torch.tensor(0.0, device=device)
        total_union = torch.tensor(0.0, device=device)

        for images, masks, _, sep_weights in dataloader:     # AGORA 4 itens
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            sep_weights = sep_weights.to(device, non_blocking=True)
            optimizer.zero_grad()

            prediction = model(images)

            probs = F.softmax(prediction, dim=1)
            ce_per_pixel = criterion(prediction, masks)          # (B, H, W), sem reduzir
            loss = (ce_per_pixel * sep_weights).mean() \
                + dice_weight_interior * dice_loss(probs, masks, 1) \
                + dice_weight_boundary * dice_loss(probs, masks, 2)
            
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_class = torch.argmax(prediction, dim=1)
                intersection = ((pred_class == 1) & (masks == 1)).float().sum()
                union = (pred_class == 1).float().sum() + (masks == 1).float().sum() - intersection

                accumulated_loss += loss.detach()
                total_intersection += intersection
                total_union += union

        epoch_loss = accumulated_loss.item() / num_batches
        ti = total_intersection.item()
        tu = total_union.item()

        epoch_iou = ti / (tu + 1e-6)
        epoch_dice = (2.0 * ti) / (tu + ti + 1e-6)

        history['loss'].append(epoch_loss)
        history['iou'].append(epoch_iou)
        history['dice'].append(epoch_dice)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}")

    return history


def semantic_to_instances(
    pred_logits: torch.Tensor,
    min_blob_px: int = 3,
    min_peak_distance: int = 2,
    smooth_sigma: float = 0.5,   # suaviza a prob. antes de buscar picos
) -> np.ndarray:
    probs = torch.softmax(pred_logits, dim=0).cpu().numpy()
    pred_class = np.argmax(probs, axis=0)

    interior_mask = (pred_class == 1)
    foreground_mask = (pred_class == 1) | (pred_class == 2)

    # Caminho principal (igual ao que já funcionava): componentes do
    # interior classificado, erodidos individualmente, com fallback
    # de 1 marcador por componente. Baixo risco de oversegmentation.
    raw_labels, num_raw = ndi.label(interior_mask)

    markers = np.zeros_like(pred_class, dtype=np.int32)
    next_id = 1

    for comp_id in range(1, num_raw + 1):
        comp_mask = (raw_labels == comp_id)
        comp_eroded = ndi.binary_erosion(comp_mask, iterations=2)
        if comp_eroded.sum() > 0:
            markers[comp_eroded] = next_id
        else:
            ys, xs = np.nonzero(comp_mask)
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            if not comp_mask[cy, cx]:
                dists = (ys - cy) ** 2 + (xs - cx) ** 2
                idx = np.argmin(dists)
                cy, cx = ys[idx], xs[idx]
            markers[cy, cx] = next_id
        next_id += 1

    # Caminho de resgate: SÓ para blobs de foreground que não têm
    # NENHUM marcador ainda (o caso problemático de células pequenas
    # coladas sem interior classificado). Não mexe nas células que já
    # funcionavam bem via interior classificado.
    fg_labels, num_fg = ndi.label(foreground_mask)
    for fg_id in range(1, num_fg + 1):
        fg_blob = (fg_labels == fg_id)

        if fg_blob.sum() < min_blob_px:
            continue
        if np.any(markers[fg_blob] > 0):
            continue  # já resolvido pelo caminho principal

        # Suaviza a probabilidade de interior SÓ dentro deste blob antes
        # de buscar picos -- isso remove platôs artificiais e ruído fino,
        # evitando múltiplos picos espúrios na mesma mancha.
        local_prob = np.where(fg_blob, probs[1], 0.0)
        local_prob_smooth = ndi.gaussian_filter(local_prob, sigma=smooth_sigma)

        peaks = peak_local_max(
            local_prob_smooth,
            min_distance=min_peak_distance,
            labels=fg_blob,
            exclude_border=False,
        )

        if len(peaks) == 0:
            dist = ndi.distance_transform_edt(fg_blob)
            y, x = np.unravel_index(np.argmax(dist), dist.shape)
            markers[y, x] = next_id
            next_id += 1
        else:
            for (y, x) in peaks:
                markers[y, x] = next_id
                next_id += 1

    if markers.max() == 0:
        return np.zeros_like(pred_class, dtype=np.int32)

    pred_instances = watershed(probs[2, :, :], markers, mask=foreground_mask, compactness=0.1)
    return pred_instances


def calculate_instance_metrics(
    true_instances: np.ndarray,
    pred_instances: np.ndarray,
) -> Tuple[float, int, int]:
    true_ids = np.unique(true_instances)
    true_ids = true_ids[true_ids != 0]
    
    pred_ids = np.unique(pred_instances)
    pred_ids = pred_ids[pred_ids != 0]

    num_true_objects = len(true_ids)
    num_pred_objects = len(pred_ids)
    count_error = abs(num_pred_objects - num_true_objects)

    if num_true_objects == 0 and num_pred_objects == 0:
        return 1.0, count_error, num_true_objects
    if num_true_objects == 0 or num_pred_objects == 0:
        return 0.0, count_error, num_true_objects

    iou_matrix = np.zeros((num_true_objects, num_pred_objects))
    for i, t_id in enumerate(true_ids):
        t_mask = (true_instances == t_id)
        for j, p_id in enumerate(pred_ids):
            p_mask = (pred_instances == p_id)
            intersection = np.logical_and(t_mask, p_mask).sum()
            if intersection > 0:
                union = np.logical_or(t_mask, p_mask).sum()
                iou_matrix[i, j] = intersection / union
    sorted_indices = np.argsort(iou_matrix.flatten())[::-1]
    shape = iou_matrix.shape

    aps = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp = 0
        matched_true = set()
        matched_pred = set()
        for idx in sorted_indices:
            t_idx, p_idx = np.unravel_index(idx, shape)
            iou = iou_matrix[t_idx, p_idx]
            if iou < t:
                break
            if t_idx not in matched_true and p_idx not in matched_pred:
                tp += 1
                matched_true.add(t_idx)
                matched_pred.add(p_idx)
                
        fp = num_pred_objects - tp
        fn = num_true_objects - tp
        ap = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        aps.append(ap)
        
    mAP = np.mean(aps)
    return mAP, count_error, num_true_objects


def evaluate2(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k_samples: int = 6
) -> Tuple[List[float], List[int], List[int], List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    model.eval()
    all_mAPs = []
    all_count_errors = []
    all_densities = []
    samples = []

    for images, _, real_instances, _ in dataloader:
        with torch.no_grad():
            preds = model(images.to(device))
        real_instances_np = real_instances.numpy()

        for i in range(images.size(0)):
            pred_instances = semantic_to_instances(preds[i])

            mAP, count_error, density = calculate_instance_metrics(
                real_instances_np[i],
                pred_instances,
            )

            all_mAPs.append(mAP)
            all_count_errors.append(count_error)
            all_densities.append(density)

            img_plot = images[i].cpu().numpy().transpose(1, 2, 0)
            samples.append((count_error, img_plot, real_instances_np[i], pred_instances))

    return all_mAPs, all_count_errors, all_densities, samples[:k_samples]



def evaluate1(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k_samples: int = 6
) -> Tuple[List[float], List[int], List[int], List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    model.eval()
    all_mAPs = []
    all_count_errors = []
    all_densities = []
    samples = []

    for images, _, real_instances in dataloader:
        with torch.no_grad():
            binary_preds = (torch.sigmoid(model(images.to(device))) > 0.5).float().cpu().numpy()

        real_instances_np = real_instances.numpy()

        for i in range(images.size(0)):
            pred_instances = np.squeeze(binary_preds[i, 0])
            pred_instances = cv.connectedComponents(pred_instances.astype(np.uint8))[1]
            
            mAP, count_error, density = calculate_instance_metrics(
                real_instances_np[i],
                pred_instances,
            )

            all_mAPs.append(mAP)
            all_count_errors.append(count_error)
            all_densities.append(density)

            img_plot = images[i].cpu().numpy().transpose(1, 2, 0)
            samples.append((count_error, img_plot, real_instances_np[i], pred_instances))


    return all_mAPs, all_count_errors, all_densities, samples[:k_samples]