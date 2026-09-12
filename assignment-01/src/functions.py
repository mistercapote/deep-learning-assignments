import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .dataset import collate_fn_ternary
from .training import train_model_ternary
from .evaluating import evaluate_instances_ternary
from .models import UNetTernary, DeepLabTernary, SegNetTernary, ParseNetTernary, PSPNetTernary

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')


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