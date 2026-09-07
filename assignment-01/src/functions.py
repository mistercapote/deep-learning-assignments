import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from typing import Dict, List, Sequence, Tuple
from skimage.segmentation import watershed
import scipy.ndimage as ndi


def training(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    part: int,
    lr: float = 1e-4,
    num_epochs: int = 10
) -> dict:
    
    if part == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        class_weights = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

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
                if part == 1:
                    pred_bin = (prediction > 0).float()
                    intersection = (pred_bin * masks).sum()
                    union = pred_bin.sum() + masks.sum() - intersection
                else:
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


def semantic_to_instances(pred_logits: torch.Tensor) -> np.ndarray:
    probs = torch.softmax(pred_logits, dim=0).cpu().numpy()
    pred_class = np.argmax(probs, axis=0)
    
    interior_mask = (pred_class == 1)
    markers, num_features = ndi.label(interior_mask)
    
    if num_features == 0:
         return np.zeros_like(pred_class, dtype=np.int32)
    
    elevation_map = probs[2, :, :]
    foreground_mask = (pred_class == 1) | (pred_class == 2)
    
    pred_instances = watershed(elevation_map, markers, mask=foreground_mask)
    return pred_instances


def calculate_instance_metrics(
    true_instances: np.ndarray,
    pred_instances: np.ndarray,
) -> Tuple[float, int, int]:
    true_ids = np.unique(true_instances)[1:]
    pred_ids = np.unique(pred_instances)[1:]

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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    part: int,
    k_samples: int = 6,
) -> Tuple[List[float], List[int], List[int], List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    model.eval()
    all_mAPs = []
    all_count_errors = []
    all_densities = []
    samples = []

    for images, _, real_instances in dataloader:
        with torch.no_grad():
            if part == 1:
                binary_preds = (torch.sigmoid(model(images.to(device))) > 0.5).float().cpu().numpy()
            elif part == 2:
                preds = model(images.to(device))
        real_instances_np = real_instances.numpy()

        for i in range(images.size(0)):
            if part == 1:
                pred_instances = np.squeeze(binary_preds[i, 0])
                pred_instances = cv.connectedComponents(pred_instances.astype(np.uint8))[1]
            elif part == 2:
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

    if len(samples) > k_samples:
        index = np.random.choice(len(samples), size=k_samples, replace=False)
        samples = [samples[i] for i in index]

    return all_mAPs, all_count_errors, all_densities, samples