import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')
IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)
IMG_SIZE = 256


class DSB2018Dataset(Dataset):
    def __init__(self, root, img_size=IMG_SIZE):
        self.root = Path(root)
        self.ids = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.img_size = img_size

    def __len__(self):
        return len(self.ids)

    def _load_instance_masks(self, mask_dir):
        mask_files = sorted(mask_dir.glob('*.png'))
        masks = []
        for mf in mask_files:
            m = np.array(Image.open(mf).convert('L'))
            masks.append((m > 0).astype(np.uint8))
        return masks

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.root / img_id / 'images' / f'{img_id}.png'
        mask_dir = self.root / img_id / 'masks'

        image = np.array(Image.open(img_path).convert('RGB'))
        instance_masks = self._load_instance_masks(mask_dir)

        H, W = image.shape[:2]
        binary_mask = np.zeros((H, W), dtype=np.uint8)
        for m in instance_masks:
            binary_mask = np.maximum(binary_mask, m)

        image_r = cv2.resize(image, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_LINEAR)
        binary_mask_r = cv2.resize(binary_mask, (self.img_size, self.img_size),
                                    interpolation=cv2.INTER_NEAREST)

        if self.augment is not None:
            aug = self.augment(image=image_r, mask=binary_mask_r)
            image_r, binary_mask_r = aug['image'], aug['mask']

        image_t = torch.from_numpy(image_r / 255.0).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(binary_mask_r.astype(np.float32)).unsqueeze(0)

        # instâncias redimensionadas junto (mantidas separadas p/ avaliação)
        instance_masks_r = [
            cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            for m in instance_masks
        ]

        return image_t, mask_t, instance_masks_r, img_id


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    instance_masks = [b[2] for b in batch]   # lista de listas (tam. variável)
    ids = [b[3] for b in batch]
    return images, masks, instance_masks, ids


# ---------------------------------------------------------------------------
# 1. Treino — BCE + Dice loss, e IoU/Dice como métricas semânticas
# ---------------------------------------------------------------------------


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def bce_dice_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)


@torch.no_grad()
def compute_iou_dice(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)
    return iou.mean().item(), dice.mean().item()

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    history = {'train_loss': [], 'val_iou': [], 'val_dice': []}
    best_iou = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks, _, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        ious, dices = [], []
        with torch.no_grad():
            for images, masks, _, _ in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                logits = model(images)
                iou, dice = compute_iou_dice(logits, masks)
                ious.append(iou)
                dices.append(dice)
        val_iou, val_dice = np.mean(ious), np.mean(dices)
        scheduler.step(val_iou)

        history['train_loss'].append(train_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)
        print(f'  loss={train_loss:.4f}  val_IoU={val_iou:.4f}  val_Dice={val_dice:.4f}')

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pt')

    return history

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
def evaluate_instances(model, loader, prob_threshold=0.5, min_size=5, k_samples=6):
    model.eval()

    n_threshold = len(IOU_THRESHOLDS)
    tp_total = np.zeros(n_threshold)
    denom_total = np.zeros(n_threshold)

    count_errors = []
    densities = []
    samples = []

    for images, _, instance_masks_batch, _ in tqdm(loader, desc='Avaliando instâncias'):
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

            img_plot = images[i].cpu().numpy().transpose(1, 2, 0)
            samples.append((count_error, img_plot, gt_masks, pred_masks))

            if n_pred == 0 and n_gt == 0:
                continue 

            if n_pred == 0 or n_gt == 0:
                denom_total += n_pred + n_gt
                continue

            pairs = []
            for i, m1 in enumerate(pred_masks):
                for j, m2 in enumerate(gt_masks):
                    inter = np.logical_and(m1, m2).sum()
                    union = np.logical_or(m1, m2).sum()
                    pairs.append((inter / union if union != 0 else 0.0, i, j))
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

                tp_total[k] += tp
                denom_total[k] += tp + fp + fn

    aps = np.where(denom_total > 0, tp_total / denom_total, 1.0)
    mAP = np.mean(aps)
    return mAP, aps, count_errors, densities, samples[:k_samples]



# ---------------------------------------------------------------------------
# 5. Fracasso em função da densidade de objetos por imagem
# ---------------------------------------------------------------------------
