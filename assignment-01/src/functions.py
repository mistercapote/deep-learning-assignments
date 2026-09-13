import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .dataset import collate_fn_ternary
from .training import train_model_ternary
from .evaluating import evaluate_instances_ternary, decode_watershed
from .models import UNetTernary, DeepLabTernary, SegNetTernary, ParseNetTernary, PSPNetTernary
import cv2 as cv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')
IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)


def set_seed(seed: int = 42):
    """Garante controle determinístico total entre seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_ternary_ablation_configs():
    """
    Modelos ternários configurados para os Eixos 1 e 3.
    Note que U-Net, DeepLab, ParseNet e PSPNet compartilham a ResNet-34.
    """
    return {
        # EIXO 1: Mecanismos de recuperação de resolução (mesmo encoder ResNet-34)
        "Eixo1_SkipConnections (UNet)": lambda: UNetTernary(),
        "Eixo1_ASPP (DeepLab)": lambda: DeepLabTernary(),
        "Eixo1_PoolIndices (SegNet)": lambda: SegNetTernary(),

        # EIXO 3: Contexto global acoplado ao decoder
        "Eixo3_Baseline (Sem Contexto Global)": lambda: UNetTernary(),
        "Eixo3_ImagePooling (ParseNet)": lambda: ParseNetTernary(),
        "Eixo3_PyramidPooling (PSPNet)": lambda: PSPNetTernary(),
    }


def ablation(
    train_dataset,
    val_dataset,
    configs: dict = None,
    seeds: list = [42, 123],
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    dist_weight: float = 1.0,
    interior_thresh: float = 0.55,
    fg_thresh: float = 0.5,
    save_dir: str = "../models",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa ablações estritamente ternárias em 2 seeds, reportando média ± desvio.

    Métricas calculadas:
    - Val Loss (Loss combinada final do treinamento ternário: Classificação + Distância L1)
    - mAP [0.5:0.95] (Média do AP sobre os 10 limiares após Watershed Marcado)
    - AP@50 e AP@75 (Precisão média em limiares específicos de overlap)
    - Erro Absoluto de Contagem (|Pred - GT|)
    """
    os.makedirs(save_dir, exist_ok=True)

    if configs is None:
        configs = get_ternary_ablation_configs()

    raw_results = []

    for name, model_fn in configs.items():
        print(f"\n{'='*75}\n[Ablation Ternária] Configuração: {name}\n{'='*75}")

        for seed in seeds:
            print(f"\n--> Iniciando Seed: {seed}")
            set_seed(seed)

            g = torch.Generator()
            g.manual_seed(seed)

            # DataLoaders exclusivamente ternários
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn_ternary,
                generator=g,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn_ternary,
            )

            # Instanciação do modelo ternário
            model = model_fn().to(DEVICE)
            model_label = f"ternary_{name.replace(' ', '_').replace('/', '_')}_s{seed}"

            # 1. Treinamento com a função ternária nativa
            history = train_model_ternary(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                label=model_label,
                epochs=epochs,
                lr=lr,
                dist_weight=dist_weight,
            )

            # 2. Carrega os melhores pesos salvos por menor val_loss
            ckpt_path = os.path.join(save_dir, f"best_model_{model_label}.pt")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=DEVICE))

            best_val_loss = min(history['val_loss']) if len(history['val_loss']) > 0 else float('nan')

            # 3. Avaliação de Instâncias via Watershed Marcado (Função Ternária)
            aps, count_errors, _, _, _ = evaluate_instances_ternary(
                model=model,
                loader=val_loader,
                interior_thresh=interior_thresh,
                fg_thresh=fg_thresh,
                min_marker_size=5,
            )

            mAP = float(np.mean(aps))
            ap_50 = float(aps[0])
            ap_75 = float(aps[5]) if len(aps) > 5 else float(aps[-1])
            mean_count_err = float(np.mean(count_errors))

            print(f"Resultado Seed {seed} | Val Loss: {best_val_loss:.4f} | mAP: {mAP:.4f} | AP@50: {ap_50:.4f} | Erro Cont: {mean_count_err:.2f}")

            raw_results.append({
                "Configuração": name,
                "Seed": seed,
                "Val Loss": best_val_loss,
                "mAP [0.5:0.95]": mAP,
                "AP@50": ap_50,
                "AP@75": ap_75,
                "Erro Contagem": mean_count_err,
            })

    # 4. Agrupamento e formatação: média ± desvio
    df_runs = pd.DataFrame(raw_results)
    numeric_cols = ["Val Loss", "mAP [0.5:0.95]", "AP@50", "AP@75", "Erro Contagem"]
    grouped = df_runs.groupby("Configuração")

    summary_rows = []
    for name, group in grouped:
        row = {"Configuração": name}
        for col in numeric_cols:
            mean = group[col].mean()
            std = group[col].std(ddof=1) if len(group) > 1 else 0.0
            if col == "Erro Contagem" or col == "Val Loss":
                row[col] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[col] = f"{mean:.4f} ± {std:.4f}"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).set_index("Configuração")
    return df_summary, df_runs


def apply_corruption(images, corruption, intensity):
    images_np = images.detach().cpu().numpy()

    corrupted = []

    for img in images_np:

        # C,H,W -> H,W,C
        img = np.transpose(img, (1, 2, 0))

        if corruption == "blur":

            kernels = {
                1: 3,
                2: 7,
                3: 11
            }

            k = kernels[intensity]

            img_corrupted = cv.GaussianBlur(
                img,
                (k, k),
                0
            )

        elif corruption == "noise":

            stds = {
                1: 0.03,
                2: 0.08,
                3: 0.15
            }

            noise = np.random.normal(
                0,
                stds[intensity],
                img.shape
            )

            img_corrupted = img + noise

        elif corruption == "brightness_contrast":

            factors = {
                1: (1.10, 0.05),
                2: (1.25, 0.10),
                3: (1.40, 0.15)
            }

            contrast, brightness = factors[intensity]

            img_corrupted = (
                img * contrast + brightness
            )

        else:
            raise ValueError(
                f"Corrupção desconhecida: {corruption}"
            )

        img_corrupted = np.clip(
            img_corrupted,
            0,
            1
        )

        # H,W,C -> C,H,W
        corrupted.append(
            np.transpose(
                img_corrupted,
                (2, 0, 1)
            )
        )

    return torch.tensor(
        np.stack(corrupted),
        dtype=images.dtype
    )



@torch.no_grad()
def evaluate_corrupted(
    model,
    dataloader,
    device,
    corruption,
    intensity
):
    model.eval()

    n_thresholds = len(IOU_THRESHOLDS)
    tp_total = np.zeros(n_thresholds)
    denom_total = np.zeros(n_thresholds)

    count_errors = []

    for images, _, _, instance_masks_batch, _ in dataloader:

        # Aplica a corrupção
        corrupted_images = apply_corruption(
            images,
            corruption,
            intensity
        ).to(device)

        # Predição do modelo ternário
        logits_cls, dist_pred = model(corrupted_images)

        probs_batch = (
            F.softmax(logits_cls, dim=1)
            .cpu()
            .numpy()
        )

        dists_batch = (
            dist_pred.squeeze(1)
            .cpu()
            .numpy()
        )

        # Avalia cada imagem do batch
        for b in range(corrupted_images.size(0)):

            pred_masks = decode_watershed(
                probs_batch[b],
                dists_batch[b],
                interior_thresh=0.55,
                fg_thresh=0.5,
                min_marker_size=5
            )

            gt_masks = instance_masks_batch[b]

            n_pred = len(pred_masks)
            n_gt = len(gt_masks)

            # Erro na contagem
            count_errors.append(
                abs(n_pred - n_gt)
            )

            # Caso não haja objetos
            if n_pred == 0 and n_gt == 0:
                continue

            if n_pred == 0 or n_gt == 0:
                denom_total += n_pred + n_gt
                continue

            # Máscaras achatadas
            pred_flat = np.array(
                pred_masks,
                dtype=np.int32
            ).reshape(n_pred, -1)

            gt_flat = np.array(
                gt_masks,
                dtype=np.int32
            ).reshape(n_gt, -1)

            # Interseção
            inter = np.dot(
                pred_flat,
                gt_flat.T
            )

            # Áreas
            area_pred = pred_flat.sum(axis=1)[:, np.newaxis]
            area_gt = gt_flat.sum(axis=1)[np.newaxis, :]

            # União
            union = area_pred + area_gt - inter

            # IoU
            iou_matrix = inter / np.maximum(
                union,
                1e-6
            )

            # Possíveis pares
            pairs = [
                (iou_matrix[i, j], i, j)
                for i in range(n_pred)
                for j in range(n_gt)
                if iou_matrix[i, j] > 0
            ]

            pairs.sort(
                key=lambda x: -x[0]
            )

            # Avalia em cada threshold de IoU
            for k, threshold in enumerate(IOU_THRESHOLDS):

                matched_pred = set()
                matched_gt = set()

                tp = 0

                for iou, i, j in pairs:

                    if iou < threshold:
                        break

                    if (
                        i not in matched_pred
                        and j not in matched_gt
                    ):
                        matched_pred.add(i)
                        matched_gt.add(j)
                        tp += 1

                fp = n_pred - tp
                fn = n_gt - tp

                tp_total[k] += tp
                denom_total[k] += tp + fp + fn

    # AP para cada threshold
    aps = np.where(
        denom_total > 0,
        tp_total / denom_total,
        1.0
    )

    # mAP
    mAP = np.mean(aps)

    # Erro médio de contagem
    count_error = np.mean(count_errors)

    return mAP, count_error