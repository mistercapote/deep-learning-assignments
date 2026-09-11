import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
from pathlib import Path
import warnings
import scipy.ndimage as ndi

# def instance_to_interior_boundary(mask, interior_frac=0.5, min_interior_px=1):
#     dist = ndi.distance_transform_edt(mask)
#     if dist.max() <= 0:
#         return np.zeros_like(mask, dtype=bool), mask.astype(bool)
    
#     thresh = max(1.0, dist.max() * interior_frac)
#     interior = dist >= thresh
    
#     # fallback de segurança: garante ao menos 1 px de interior
#     if interior.sum() < min_interior_px:
#         y, x = np.unravel_index(np.argmax(dist), dist.shape)
#         interior = np.zeros_like(mask, dtype=bool)
#         interior[y, x] = True
    
#     boundary = mask.astype(bool) & (~interior)
#     return interior, boundary


def instance_to_interior_boundary(mask, erosion_px=2, min_interior_px=1):
    mask_bool = mask.astype(bool)

    if mask_bool.sum() == 0:
        return np.zeros_like(mask, dtype=bool), mask_bool

    # Erosão em N pixels fixos, em vez de fração da distância máxima.
    # Isso deixa o interior proporcionalmente mais "gordo" em núcleos
    # pequenos/alongados, onde dist.max() já era baixo.
    interior = ndi.binary_erosion(mask_bool, iterations=erosion_px)

    # fallback de segurança: garante ao menos 1 px de interior
    if interior.sum() < min_interior_px:
        dist = ndi.distance_transform_edt(mask_bool)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        interior = np.zeros_like(mask, dtype=bool)
        interior[y, x] = True

    boundary = mask_bool & (~interior)
    return interior, boundary

def compute_separation_weight_map(instance_masks, w0=10.0, sigma=5.0):
    """
    Gera um mapa de peso por pixel que aumenta perto de fronteiras entre
    instâncias vizinhas -- técnica do paper original do U-Net para forçar
    o modelo a aprender a separar núcleos próximos/tocando-se.
    """
    n_instances = instance_masks.shape[0]
    h, w = instance_masks.shape[1:]

    if n_instances < 2:
        return np.ones((h, w), dtype=np.float32)

    # Para cada instância, distância de CADA pixel até a borda dela
    # (fora da máscara -> distância até a instância mais próxima)
    dist_maps = np.zeros((n_instances, h, w), dtype=np.float32)
    for i in range(n_instances):
        dist_maps[i] = ndi.distance_transform_edt(instance_masks[i] == 0)

    # d1: distância até a instância mais próxima; d2: até a segunda mais próxima
    sorted_dists = np.sort(dist_maps, axis=0)
    d1 = sorted_dists[0]
    d2 = sorted_dists[1]

    weight_map = 1.0 + w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
    return weight_map.astype(np.float32)



def to_tensors(image: np.ndarray, instance_masks: np.ndarray, part: int = 1, espessura_fronteira: int = 1):
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
    
    img_h, img_w = instance_masks.shape[1:]
    instance_gt = np.zeros((img_h, img_w), dtype=np.int32)
    
    if part == 1:
        binary_mask = (instance_masks.sum(axis=0) > 0).astype(np.float32)
        target_tensor = torch.from_numpy(binary_mask).unsqueeze(0)
        
        for i, mask in enumerate(instance_masks):
            instance_gt[mask > 0] = i + 1
            
    elif part == 2:
        semantic_target = np.zeros((img_h, img_w), dtype=np.int64)

        for i, mask in enumerate(instance_masks):
            if mask.sum() == 0:
                continue
            
            # CORREÇÃO AQUI: Substituir o 2 fixo pela variável do hiperparâmetro
            interior, boundary = instance_to_interior_boundary(mask, erosion_px=espessura_fronteira)
            
            semantic_target[interior] = 1
            semantic_target[boundary] = 2
            instance_gt[mask > 0] = i + 1

        target_tensor = torch.from_numpy(semantic_target).long()
        weight_map = compute_separation_weight_map(instance_masks)          
        weight_tensor = torch.from_numpy(weight_map)                        

    if part == 1:
        return image_tensor, target_tensor, instance_gt
    else:
        return image_tensor, target_tensor, instance_gt, weight_tensor




class DSB2018Dataset(Dataset):
    def __init__(self, root_dir: str, img_size: int, part: int, sample_ids: list = None,
                 cache_in_memory: bool = True, cache_in_disk: bool = True,
                 espessura_fronteira: int = 1):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.sample_ids = sample_ids or sorted(
            p.name for p in self.root_dir.iterdir() if p.is_dir()
        )
        self.cache_in_memory = cache_in_memory
        self.cache_in_disk = cache_in_disk
        self.memory_cache = {}  

        if self.cache_in_memory and len(self.sample_ids) > 1000:
            warnings.warn("Dataset grande (>1000 samples). O cache em memória pode esgotar a RAM.")

        self.sample_paths = []
        for sid in self.sample_ids:
            sdir = self.root_dir / sid
            img_path = next((sdir / "images").glob("*.png"), None)
            mask_paths = sorted((sdir / "masks").glob("*.png"))
            self.sample_paths.append((img_path, mask_paths, sdir))

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self.memory_cache:
            return self.memory_cache[idx]

        img_path, mask_paths, sample_dir = self.sample_paths[idx]
        
        disk_cache_path = sample_dir / f"cache_part{self.part}_{self.img_size}_esp{self.espessura_fronteira}.pt"
        if self.cache_in_disk and disk_cache_path.exists():
            tensors = torch.load(disk_cache_path, weights_only=False)
            if self.cache_in_memory:
                self.memory_cache[idx] = tensors
            return tensors

        image = cv.imread(str(img_path), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (self.img_size, self.img_size), interpolation=cv.INTER_LINEAR)

        instance_masks = np.zeros((len(mask_paths), self.img_size, self.img_size), dtype=np.uint8)
        for i, mask_path in enumerate(mask_paths):
            mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST)
            instance_masks[i] = (mask > 127).astype(np.uint8)

        tensors = to_tensors(image, instance_masks, self.part, self.espessura_fronteira)

        if self.cache_in_disk:
            torch.save(tensors, disk_cache_path)
        if self.cache_in_memory:
            self.memory_cache[idx] = tensors

        return tensors

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


import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn import functional as F


class UNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d3 = self.dec3(torch.cat([self.up3(x4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))

        return self.final_conv(self.up0(d1))


class UNetTernary(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU())
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.segmentation_head = nn.Conv2d(32, 3, kernel_size=1)
        self.distance_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d3 = self.dec3(torch.cat([self.up3(x4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        d0 = self.up0(d1)

        logits_class = self.segmentation_head(d0)
        dist_pred = self.distance_head(d0)  
        return logits_class, dist_pred