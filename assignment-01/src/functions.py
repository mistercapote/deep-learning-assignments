import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN
from .models import   ParseNetDDimensional, PSPNetDDimensional, UNetTernary
from .training import train_model_ternary
from .evaluating import evaluate_instances_ternary








import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# Importações dos módulos do projeto
from models import (
    ASPP,
    UNetDDimensional,
    PSPNetDDimensional,
    ParseNetDDimensional,
    SegNetDDimensional,
    DeepLabDDimensional,
    DeepLabResNet18
)
from training import train_model_binary, compute_iou_dice
from evaluating import evaluate_instances_binary

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42):
    """Garante reprodutibilidade estrita entre execuções."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelBinaryWrapper(nn.Module):
    """
    Garante que modelos que retornam tuplas (ex: semantic_head, embed_head)
    forneçam apenas os logits de canal único (B, 1, H, W) esperados
    pelas rotinas train_model_binary e evaluate_instances_binary.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def get_ablation_models() -> dict:
    """
    Mapeamento de modelos para os Eixos 1 e 3.
    Todos os modelos baseados em ResNet compartilham o encoder ResNet-18.
    """
    return {
        # Eixo 1: Mecanismos de recuperação de resolução (mesmo encoder ResNet-18)
        "Eixo1_SkipConnections (U-Net)": lambda: UNetDDimensional(D=1),
        "Eixo1_ASPP (DeepLab)": lambda: DeepLabResNet18(D=1),
        "Eixo1_PoolIndices (SegNet)": lambda: SegNetDDimensional(D=1),

        # Eixo 3: Contexto global acoplado ao decoder
        # A U-Net serve de baseline (sem pooling global adicional)
        "Eixo3_Baseline (Sem Contexto)": lambda: UNetDDimensional(D=1),
        "Eixo3_ImagePooling (ParseNet)": lambda: ParseNetDDimensional(D=1),
        "Eixo3_PyramidPooling (PSPNet)": lambda: PSPNetDDimensional(D=1),
    }


def ablation(
    train_dataset,
    val_dataset,
    configs: dict = None,
    seeds: list = [42, 123],
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    save_dir: str = "../models",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa o protocolo experimental de ablações reportando média ± desvio.

    Parâmetros:
        train_dataset: Dataset de treino (formato binário).
        val_dataset: Dataset de validação (com instâncias para matching).
        configs: Dicionário {nome_config: factory_function_modelo}.
                 Se None, executa as arquiteturas padrão dos Eixos 1 e 3.
        seeds: Lista com 2 seeds para execução estocástica.
        epochs: Número de épocas de treinamento por seed.
        batch_size: Tamanho do mini-batch.
        lr: Taxa de aprendizado.
        save_dir: Diretório para salvar pesos checkpoints.

    Retorna:
        df_summary: Tabela agregada com métricas no formato 'média ± desvio'.
        df_runs: Tabela discriminada contendo os valores brutos de cada seed.
    """
    os.makedirs(save_dir, exist_ok=True)

    if configs is None:
        configs = get_ablation_models()

    from dataset import collate_fn_binary
    raw_results = []

    for name, model_fn in configs.items():
        print(f"\n{'='*70}\nIniciando Configuração: {name}\n{'='*70}")

        for seed in seeds:
            print(f"--> Executando Seed {seed}...")
            set_seed(seed)

            # DataLoaders com gerador determinístico
            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn_binary,
                generator=g,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn_binary,
            )

            # Instanciação e envelopamento para compatibilidade
            model = ModelBinaryWrapper(model_fn()).to(DEVICE)
            model_label = f"ablation_{name.replace(' ', '_').replace('/', '_')}_seed{seed}"

            # 1. Treinamento
            history = train_model_binary(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                label=model_label,
                epochs=epochs,
                lr=lr,
            )

            # Carrega o melhor checkpoint salvo durante o treino
            ckpt_path = os.path.join(save_dir, f"best_model_{model_label}.pt")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, weights_only=True))

            # 2. Avaliação de Métricas Semânticas (Nível de Pixel)
            model.eval()
            total_iou = torch.tensor(0.0, device=DEVICE)
            total_dice = torch.tensor(0.0, device=DEVICE)
            with torch.no_grad():
                for images, masks, _, _ in val_loader:
                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)
                    logits = model(images)
                    b_iou, b_dice = compute_iou_dice(logits, masks)
                    total_iou += b_iou
                    total_dice += b_dice

            val_iou = (total_iou / len(val_dataset)).item() * 100.0
            val_dice = (total_dice / len(val_dataset)).item() * 100.0

            # 3. Avaliação de Instâncias (mAP e Erro de Contagem)
            aps, count_errors, _, per_image_aps, _ = evaluate_instances_binary(
                model=model,
                loader=val_loader,
                prob_threshold=0.5,
                min_size=5,
            )

            mAP = float(np.mean(aps))
            ap_50 = float(aps[0])
            ap_75 = float(aps[5]) if len(aps) > 5 else float(aps[-1])
            mean_count_err = float(np.mean(count_errors))

            raw_results.append({
                "Configuração": name,
                "Seed": seed,
                "Val IoU (%)": val_iou,
                "Val Dice (%)": val_dice,
                "mAP [0.5:0.95]": mAP,
                "AP@50": ap_50,
                "AP@75": ap_75,
                "Erro Contagem": mean_count_err,
            })

    df_runs = pd.DataFrame(raw_results)

    # 4. Agregação em Média ± Desvio Padrão
    numeric_cols = [
        "Val IoU (%)", "Val Dice (%)",
        "mAP [0.5:0.95]", "AP@50", "AP@75", "Erro Contagem"
    ]
    grouped = df_runs.groupby("Configuração")

    summary_rows = []
    for name, group in grouped:
        row = {"Configuração": name}
        for col in numeric_cols:
            mean = group[col].mean()
            std = group[col].std(ddof=1) if len(group) > 1 else 0.0
            fmt = f"{mean:.2f} ± {std:.2f}" if "IoU" in col or "Dice" in col or "Contagem" in col else f"{mean:.4f} ± {std:.4f}"
            row[col] = fmt
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows).set_index("Configuração")
    return df_summary, df_runs



# def ablation(dataloader_train, dataloader_val, device, axis, seeds: list[int] = [42, 100]):
    
#     architectures = {
#             "Parse": ParseNetDDimensional,
#             "UNeT": UNetTernary,
#             "PSP": PSPNetDDimensional
#         }

#     maP_result_comb_1 = {}
#     maP_result_comb_2 = {}
   
#     for name, model_class in architectures.items():
#         mAP_results_1 = []
#         mAP_results_2 = []
#         for current_seed in seeds:
#             print(f"Avaliando Arquitetura: {name} com seed {current_seed}")
#             torch.manual_seed(current_seed)
#             np.random.seed(current_seed)
#             model = model_class().to(device)
#             train_model_ternary(model, dataloader_train, dataloader_val)
#             aps, _, _, per_image_aps, _ = evaluate_instances_ternary(model, dataloader_val, device, part=2)
#             mAP_results_1.append(np.mean(aps))
#             mAP_results_2.append(np.mean(per_image_aps))
#         maP_result_comb_1[name] = (np.mean(mAP_results_1), np.std(mAP_results_1))
#         maP_result_comb_2[name] = (np.mean(mAP_results_2), np.std(mAP_results_2))
    
#     return maP_result_comb_1, maP_result_comb_2 

def create_mosaic_real(dataset, indices=[0, 1, 2, 3]):
    """
    Cria uma imagem grande (mosaico 2x2) combinando 4 amostras do dataset real.
    """
    # Pega a primeira amostra para checar as dimensões e o formato (PyTorch [C, H, W] ou NumPy)
    sample_img, sample_mask, sample_inst = dataset[indices[0]]
    
    # Converte para numpy caso estejam em tensor do PyTorch
    if hasattr(sample_img, "detach"):
        sample_img = sample_img.cpu().numpy()
    if hasattr(sample_inst, "detach"):
        sample_inst = sample_inst.cpu().numpy()
        
    C, H, W = sample_img.shape
    
    # Cria os arrays vazios para o mosaico (2x o tamanho original em H e W)
    mosaic_img = np.zeros((C, H * 2, W * 2), dtype=sample_img.dtype)
    mosaic_inst = np.zeros((H * 2, W * 2), dtype=np.int32)
    
    # Posições do grid 2x2: ( y_offset, x_offset, indice_no_dataset )
    positions = [
        (0, 0, indices[0]),
        (0, W, indices[1]),
        (H, 0, indices[2]),
        (H, W, indices[3])
    ]
    
    max_inst_id = 0
    
    for y_offset, x_offset, idx in positions:
        img, _, inst = dataset[idx]
        
        if hasattr(img, "detach"):
            img = img.cpu().numpy()
        if hasattr(inst, "detach"):
            inst = inst.cpu().numpy()
            
        # Insere a imagem no quadrante correspondente
        mosaic_img[:, y_offset:y_offset+H, x_offset:x_offset+W] = img
        
        # Ajusta os IDs das instâncias para que não se sobreponham entre os 4 quadrantes
        inst_shifted = inst.copy()
        valid_mask = inst > 0
        inst_shifted[valid_mask] += max_inst_id
        
        if valid_mask.any():
            max_inst_id = inst_shifted.max()
            
        mosaic_inst[y_offset:y_offset+H, x_offset:x_offset+W] = inst_shifted
        
    return mosaic_img, mosaic_inst


def mosaic_tile_inference_demo(mosaic_img, tile_size=128, overlap=32):
    """
    Simula o fatiamento de uma imagem grande em tiles com sobreposição (tiling),
    revelando o desafio das fronteiras para objetos divididos entre os blocos.
    """
    _, H, W = mosaic_img.shape
    stride = tile_size - overlap
    
    tiles = []
    coordinates = []
    
    for y in range(0, H - overlap, stride):
        for x in range(0, W - overlap, stride):
            # Garante que o tile não ultrapasse as bordas da imagem grande
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            tile = mosaic_img[:, y_start:y_end, x_start:x_end]
            tiles.append(tile)
            coordinates.append((y_start, y_end, x_start, x_end))
            
    return tiles, coordinates


def mosaic_inference_with_fusion(model, large_image,part=1,  tile_size=128, overlap=32, device='cpu'):
    """
    Realiza inferência em mosaico (tiles) em uma imagem grande com sobreposição
    e aplica fusão de instâncias nas bordas dos tiles baseada em intersecção.
    """
    # Se o modelo não for nulo, garantimos que está em modo de avaliação
    if model is not None:
        model.eval()
        
    C, H, W = large_image.shape
    stride = tile_size - overlap
    
    # Canvas para acumular as predições finais
    global_instances = np.zeros((H, W), dtype=np.int32)
    next_instance_id = 1
    
    for y in range(0, H - overlap, stride):
        for x in range(0, W - overlap, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            # Recorta o tile da imagem
            tile = large_image[:, y_start:y_end, x_start:x_end]
            tile_tensor = torch.tensor(tile).unsqueeze(0).float().to(device)
                
            with torch.no_grad():
                # Faz a predição local do tile
                if part ==1:
                    pred_bin = model(tile_tensor)
                    binary_pred = (torch.sigmoid(model(tile_tensor)) > 0.5).float().cpu().numpy()
                    pred_mask = np.squeeze(binary_pred[0, 0]).astype(np.uint8)
                    
                # Extrai instâncias locais do tile
            
                    _, tile_insts = cv.connectedComponents(pred_mask)
                elif part ==2:
                    pred_bin, pred_emb = model(tile_tensor)
                    
                    # Decodificação da Trilha B (DBSCAN) usando a função que você criou
                    tile_insts = embeddings_to_instances(pred_bin[0], pred_emb[0])
                    tile_insts = np.squeeze(tile_insts) # Garante que seja 2D (H,W)

            # Reatribui IDs globais e resolve conflitos na sobreposição
            for l_id in np.unique(tile_insts)[1:]:
                local_mask = (tile_insts == l_id)
                
                # Olhamos para a região correspondente no canvas global
                existing_canvas_region = global_instances[y_start:y_end, x_start:x_end]
                
                # Pega os IDs e a quantidade de pixels de intersecção de cada um
                overlap_ids, counts = np.unique(existing_canvas_region[local_mask], return_counts=True)
                
                # Filtra o fundo (ID 0)
                valid_mask = overlap_ids > 0
                overlap_ids = overlap_ids[valid_mask]
                counts = counts[valid_mask]

                if len(overlap_ids) > 0:
                    # Pega o ID global com a maior quantidade de pixels sobrepostos
                    best_match_idx = np.argmax(counts)
                    g_id = overlap_ids[best_match_idx]
                    
                    t_mask = (existing_canvas_region == g_id)
                    
                    intersection = counts[best_match_idx] # Já temos a área de intersecção do argmax!
                    min_area = min(t_mask.sum(), local_mask.sum())
                    
                    if min_area > 0 and (intersection / min_area) > 0.5:
                        global_instances[y_start:y_end, x_start:x_end][local_mask] = g_id
                    else:
                        global_instances[y_start:y_end, x_start:x_end][local_mask] = next_instance_id
                        next_instance_id += 1
                else:
                    global_instances[y_start:y_end, x_start:x_end][local_mask] = next_instance_id
                    next_instance_id += 1
                    
    return global_instances


def mosaic_inference_naive(model, large_image, tile_size=128, overlap=32, device='cpu'):
    """ Inferência ingênua (sem correção) para mostrar o erro na fronteira. """
    if model is not None: model.eval()
    C, H, W = large_image.shape
    stride = tile_size - overlap
    global_instances = np.zeros((H, W), dtype=np.int32)
    next_instance_id = 1
    
    for y in range(0, H - overlap, stride):
        for x in range(0, W - overlap, stride):
            y_end, x_end = min(y + tile_size, H), min(x + tile_size, W)
            y_start, x_start = max(0, y_end - tile_size), max(0, x_end - tile_size)
            
            tile = large_image[:, y_start:y_end, x_start:x_end]
            tile_tensor = torch.tensor(tile).unsqueeze(0).float().to(device)
                
            with torch.no_grad():
                binary_pred = (torch.sigmoid(model(tile_tensor)) > 0.5).float().cpu().numpy()
                pred_mask = np.squeeze(binary_pred[0, 0]).astype(np.uint8)
                
            _, tile_insts = cv.connectedComponents(pred_mask)
            
            # Abordagem ingênua: Apenas sobrescreve o canvas sem verificar se o objeto já existe
            for l_id in np.unique(tile_insts)[1:]:
                local_mask = (tile_insts == l_id)
                global_instances[y_start:y_end, x_start:x_end][local_mask] = next_instance_id
                next_instance_id += 1
                    
    return global_instances
