import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader
from typing import Dict, List, Sequence, Tuple
from .models import DeepLabDDimensional, SegNetDDimensional, UNetDDimensional, ParseNetDDimensional, PSPNetDDimensional



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex  # Necessário para XPU

def training(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    part: int,
    lr: float = 1e-4,
    num_epochs: int = 10,
    use_amp: bool = True,
    compile_model: bool = False,
    set_to_none: bool = True,
    use_ipex_optimize: bool = True,  # Novo: otimização IPEX para XPU/CPU
) -> dict:
    """
    Treina o modelo para parte 1 (binária) ou parte 2 (embeddings).
    Suporte nativo para CUDA, Intel XPU (Arc) e CPU.
    Otimizado com AMP, compilação dinâmica e zero_grad eficiente.
    """
    
    model.to(device)
    
    # --- 1. Otimização específica para Intel (IPEX) ---
    if use_ipex_optimize and device.type in ('xpu', 'cpu'):
        try:
            # Converte para channels_last (melhor para conv2d no Intel)
            model = model.to(memory_format=torch.channels_last)
            # Otimiza o modelo para a arquitetura Intel
            model = ipex.optimize(model, dtype=torch.bfloat16 if use_amp else torch.float32)
            print("✅ IPEX optimize aplicado ao modelo")
        except Exception as e:
            print(f"⚠️ IPEX optimize falhou: {e}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Otimiza o optimizer também (se IPEX disponível)
    if use_ipex_optimize and device.type in ('xpu', 'cpu'):
        try:
            optimizer = ipex.optimize(optimizer, dtype=torch.bfloat16 if use_amp else torch.float32)
            print("✅ IPEX optimize aplicado ao optimizer")
        except Exception as e:
            print(f"⚠️ IPEX optimize falhou no optimizer: {e}")

    num_batches = len(dataloader)
    
    # --- 2. Configuração do AMP (Mixed Precision) ---
    scaler = None
    amp_device_type = None
    if use_amp:
        if device.type == 'cuda':
            scaler = torch.amp.GradScaler(device_type='cuda')
            amp_device_type = 'cuda'
            print("✅ AMP (CUDA) ativado")
        elif device.type == 'xpu':
            # XPU suporta AMP nativamente via IPEX
            scaler = torch.amp.GradScaler(device_type='xpu')
            amp_device_type = 'xpu'
            print("✅ AMP (XPU) ativado")
        else:
            # CPU: usa bfloat16 (mais estável que float16)
            amp_device_type = 'cpu'
            print("✅ AMP (CPU com bfloat16) ativado")
    else:
        amp_device_type = 'cpu' if device.type == 'cpu' else device.type

    # --- 3. Compilação do modelo (PyTorch 2.0+) ---
    if compile_model:
        try:
            # 'eager' é mais estável para XPU; 'inductor' para CUDA/CPU
            backend = 'eager' if device.type == 'xpu' else 'inductor'
            model = torch.compile(model, backend=backend)
            print(f"✅ Modelo compilado com torch.compile (backend={backend})")
        except Exception as e:
            print(f"⚠️ torch.compile falhou: {e}")

    print(f"🎯 Treinando no dispositivo: {device}")
    print(f"📦 Total de batches por época: {num_batches}")

    history = {'loss': [], 'iou': [], 'dice': []}

    for epoch in range(num_epochs):
        model.train()
        
        # Acumuladores no device (evita sincronizações CPU-GPU desnecessárias)
        accumulated_loss = torch.tensor(0.0, device=device)
        total_intersection = torch.tensor(0.0, device=device)
        total_union = torch.tensor(0.0, device=device)

        for images, masks, instance_gt in dataloader:
            # non_blocking=True é seguro para XPU e CUDA
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=set_to_none)
            
            # Forward pass com AMP (autocast)
            with torch.autocast(
                device_type=amp_device_type,
                enabled=scaler is not None,
                dtype=torch.bfloat16 if device.type == 'cpu' else torch.float16
            ):
                if part == 1:
                    prediction_bin = model(images)
                    loss = criterion(prediction_bin, masks)
                else:
                    instance_gt = instance_gt.to(device, non_blocking=True)
                    prediction_bin, prediction_emb = model(images)
                    loss_bin = criterion(prediction_bin, masks)
                    loss_emb = discriminative_loss(prediction_emb, instance_gt)
                    loss = loss_bin + loss_emb

            # Backward com ou sem AMP
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Cálculo das métricas otimizado (prediction_bin > 0 equivale a sigmoid > 0.5)
            with torch.no_grad():
                binary_prediction = (prediction_bin > 0).float()
                intersection = (binary_prediction * masks).sum()
                union = binary_prediction.sum() + masks.sum() - intersection

                accumulated_loss += loss.detach()
                total_intersection += intersection
                total_union += union

        # Cálculo das médias da época
        epoch_loss = accumulated_loss.item() / num_batches
        ti = total_intersection.item()
        tu = total_union.item()
        
        epoch_iou = ti / (tu + 1e-6)
        epoch_dice = (2.0 * ti) / (tu + ti + 1e-6)
        
        history['loss'].append(epoch_loss)
        history['iou'].append(epoch_iou)
        history['dice'].append(epoch_dice)

        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}")

    return history




# def training(
#     model: nn.Module,
#     dataloader: DataLoader,
#     device: torch.device,
#     part: int,
#     lr: float = 1e-4,
#     num_epochs: int = 10,
#     use_amp: bool = True,
#     compile_model: bool = False,
#     set_to_none: bool = True,
# ) -> dict:
#     """
#     Treina o modelo para parte 1 (binária) ou parte 2 (embeddings).
#     Otimizado com AMP (Mixed Precision), compilação dinâmica e zero_grad eficiente.
#     """
    
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     num_batches = len(dataloader)
    
#     # 1. Compilação do modelo (PyTorch 2.0+) - funde operações e otimiza o grafo
#     if compile_model:
#         try:
#             model = torch.compile(model)
#             print("Modelo compilado com torch.compile")
#         except Exception as e:
#             print(f"torch.compile não disponível: {e}")
    
#     # 2. AMP (Mixed Precision) - usa Tensor Cores da GPU e reduz memória
#     scaler = None
#     if use_amp and device.type == 'cuda':
#         scaler = torch.cuda.amp.GradScaler()
#         print("AMP (Mixed Precision) ativado")
#     else:
#         if use_amp:
#             print("AMP ignorado (dispositivo não é CUDA)")

#     print(f"Treinando no dispositivo: {device}")
#     print(f"Total de batches por época: {num_batches}")

#     history = {'loss': [], 'iou': [], 'dice': []}

#     for epoch in range(num_epochs):
#         model.train()
        
#         # Acumuladores no device (evita sincronizações CPU-GPU desnecessárias por batch)
#         accumulated_loss = torch.tensor(0.0, device=device)
#         total_intersection = torch.tensor(0.0, device=device)
#         total_union = torch.tensor(0.0, device=device)

#         for images, masks, instance_gt in dataloader:
#             # non_blocking=True permite sobrepor transferência de dados com computação
#             images = images.to(device, non_blocking=True)
#             masks = masks.to(device, non_blocking=True)

#             # set_to_none reduz o custo de memória e é mais rápido que setar para 0
#             optimizer.zero_grad(set_to_none=set_to_none)
            
#             # 3. Forward pass com AMP (autocast)
#             with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=scaler is not None):
#                 if part == 1:
#                     prediction_bin = model(images)
#                     loss = criterion(prediction_bin, masks)
#                 else:
#                     instance_gt = instance_gt.to(device, non_blocking=True)
#                     prediction_bin, prediction_emb = model(images)
#                     loss_bin = criterion(prediction_bin, masks)
#                     loss_emb = discriminative_loss(prediction_emb, instance_gt)
#                     loss = loss_bin + loss_emb

#             # Backward com ou sem escalonamento do AMP
#             if scaler is not None:
#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 loss.backward()
#                 optimizer.step()

#             # 4. Cálculo das métricas otimizado
#             with torch.no_grad():
#                 # Otimização: sigmoid(x) > 0.5 é matematicamente idêntico a x > 0
#                 # Isso elimina uma chamada cara ao sigmoid e uma comparação extra
#                 binary_prediction = (prediction_bin > 0).float()
#                 intersection = (binary_prediction * masks).sum()
#                 union = binary_prediction.sum() + masks.sum() - intersection

#                 accumulated_loss += loss.detach()
#                 total_intersection += intersection
#                 total_union += union

#         # Cálculo das médias da época
#         epoch_loss = accumulated_loss.item() / num_batches
#         ti = total_intersection.item()
#         tu = total_union.item()
        
#         epoch_iou = ti / (tu + 1e-6)
#         epoch_dice = (2.0 * ti) / (tu + ti + 1e-6)
        
#         history['loss'].append(epoch_loss)
#         history['iou'].append(epoch_iou)
#         history['dice'].append(epoch_dice)

#         print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}")

#     return history


# def training(
#     model: nn.Module,
#     dataloader: DataLoader,
#     device: torch.device,
#     part: int,
#     lr: float = 1e-4,
#     num_epochs: int = 10,
# ) -> dict:
#     """
#     Treina o modelo para parte 1 (binária) ou parte 2 (embeddings).
#     Retorna dicionário com histórico de loss, IoU e Dice por época.
#     """
    
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     num_batches = len(dataloader)
    
#     print(f"Treinando no dispositivo: {device}")
#     print(f"Total de batches por época: {num_batches}")

#     history = {'loss': [], 'iou': [], 'dice': []}

#     for epoch in range(num_epochs):
#         model.train()
        
#         # Acumuladores
#         accumulated_loss = torch.tensor(0.0, device=device)
#         total_intersection = torch.tensor(0.0, device=device)
#         total_union = torch.tensor(0.0, device=device)

#         for images, masks, instance_gt in dataloader:
#             images = images.to(device)
#             masks = masks.to(device)

#             if part == 1:
#                 optimizer.zero_grad()
#                 prediction_bin = model(images)
#                 loss = criterion(prediction_bin, masks)

#             elif part == 2:
#                 instance_gt = instance_gt.to(device)
#                 optimizer.zero_grad()
#                 prediction_bin, prediction_emb = model(images)
#                 loss_bin = criterion(prediction_bin, masks)
#                 loss_emb = discrimative_loss(prediction_emb, instance_gt)
#                 loss = loss_bin + loss_emb

#             loss.backward()
#             optimizer.step()

#             with torch.no_grad():
#                 binary_prediction = (torch.sigmoid(prediction_bin) > 0.5).float()
#                 intersection = (binary_prediction * masks).sum()
#                 union = binary_prediction.sum() + masks.sum() - intersection

#                 accumulated_loss += loss.detach()
#                 total_intersection += intersection
#                 total_union += union

#         # Cálculo das médias da época
#         epoch_loss = accumulated_loss.item() / num_batches
#         ti = total_intersection.item()
#         tu = total_union.item()
        
#         epoch_iou = ti / (tu + 1e-6)
#         epoch_dice = (2.0 * ti) / (tu + ti + 1e-6)
        
#         # Armazena histórico
#         history['loss'].append(epoch_loss)
#         history['iou'].append(epoch_iou)
#         history['dice'].append(epoch_dice)

#         print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}")

#     return history



def calculate_instance_metrics(
    true_instances: np.ndarray,
    pred_instances: np.ndarray,
) -> Tuple[float, int, int]:
    true_ids = np.unique(true_instances)[1:] # Remove the background label (0)
    pred_ids = np.unique(pred_instances)[1:] # Remove the background label (0)

    # Absolute count error
    num_true_objects = len(true_ids)
    num_pred_objects = len(pred_ids)
    count_error = abs(num_pred_objects - num_true_objects)

    # Extreme cases: no objects in either true or predicted masks
    if num_true_objects == 0 and num_pred_objects == 0:
        return 1.0, count_error, num_true_objects
    if num_true_objects == 0 or num_pred_objects == 0:
        return 0.0, count_error, num_true_objects

    # IoU Matrix Calculation
    iou_matrix = np.zeros((num_true_objects, num_pred_objects))
    for i, t_id in enumerate(true_ids):
        t_mask = (true_instances == t_id)
        for j, p_id in enumerate(pred_ids):
            p_mask = (pred_instances == p_id)
            intersection = np.logical_and(t_mask, p_mask).sum()
            if intersection > 0:
                union = np.logical_or(t_mask, p_mask).sum()
                iou_matrix[i, j] = intersection / union
    sorted_indices = np.argsort( iou_matrix.flatten())[::-1]
    shape = iou_matrix.shape

    #  Calculate Average Precision 
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
                tp += 1 # só os diferentes
                matched_true.add(t_idx)
                matched_pred.add(p_idx)
                
        fp = num_pred_objects - tp
        fn = num_true_objects - tp
        ap = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        aps.append(ap)
        
    mAP = np.mean(aps)
    
    return mAP, count_error, num_true_objects


def embeddings_to_instances(
    pred_bin: torch.Tensor,
    pred_emb: torch.Tensor,
    eps: float = 0.5,
    min_samples: int = 10,
) -> np.ndarray:
    mask = (torch.sigmoid(pred_bin[0]) > 0.5).cpu().numpy()
    pred_instances = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return pred_instances

    emb_foreground = (pred_emb[:, mask].T).cpu().numpy()

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(emb_foreground) + 1
    pred_instances[mask] = labels

    return pred_instances


# def train_model(
#     model: nn.Module,
#     dataloader: DataLoader,
#     device: torch.device,
#     part: int,
#     lr: float = 1e-3,
#     num_epochs: int = 5,
# ) -> None:
#     model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     num_batchs = len(dataloader)

#     for epoch in range(num_epochs):
#         model.train()
#         accumulated_loss = 0.0
#         total_intersection = 0.0
#         total_union = 0.0

#         for images, masks, instance_gt in dataloader:
#             images = images.to(device)
#             masks = masks.to(device)

#             if part == 1:
#                 optimizer.zero_grad()
#                 prediction_bin = model(images)
#                 loss = criterion(prediction_bin, masks)
#             elif part == 2:
#                 instance_gt = instance_gt.to(device)
#                 optimizer.zero_grad()
#                 prediction_bin, prediction_emb = model(images)
#                 loss_bin = criterion(prediction_bin, masks)
#                 loss_emb = discrimative_loss(prediction_emb, instance_gt)
#                 loss = loss_bin + loss_emb
#             loss.backward()
#             optimizer.step()

#             binary_prediction = (torch.sigmoid(prediction_bin) > 0.5).float()
#             intersection = (binary_prediction * masks).sum()
#             union = binary_prediction.sum() + masks.sum() - intersection

#             accumulated_loss += loss.item()
#             total_intersection += intersection.item()
#             total_union += union.item()

#         epoch_iou = total_intersection / (total_union + 1e-6)
#         epoch_dice = (2.0 * total_intersection) / (total_union + total_intersection + 1e-6)
#         print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {accumulated_loss / num_batchs:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}")





def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    part: int,
    k_samples: int = 4,
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
                binary_preds, embed_preds = model(images.to(device))
        real_instances_np = real_instances.numpy()

        for i in range(images.size(0)):
            if part == 1:
                pred_instances = np.squeeze(binary_preds[i, 0])
                pred_instances = cv.connectedComponents(pred_instances.astype(np.uint8))[1]
            elif part == 2:
                pred_instances = embeddings_to_instances(binary_preds[i], embed_preds[i])

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


def discrimative_loss(
    prediction: torch.Tensor,
    instance: torch.Tensor,
    delta_v: float = 0.1,  # Margem de variância (puxa para o centroide até esse limite)
    delta_d: float = 1.5,  # Margem de distância (empurra centroides diferentes)
) -> torch.Tensor:
    batch_size = prediction.size(0)
    total_loss = prediction.sum() * 0.0

    for b in range(batch_size):
        pred_b = prediction[b]
        inst_b = instance[b]
        objects_ids = torch.unique(inst_b)[1:]
        var_loss = 0.0
        reg_loss = 0.0
        dist_loss = 0.0
        centroids = []

        for obj_id in objects_ids:
            mask = inst_b == obj_id
            pixels = pred_b[:, mask]

            avg = torch.mean(pixels, dim=1)
            
            # Adição do epsilon (1e-6) para evitar instabilidade (gradiente NaN em 0)
            # Elevação ao quadrado e margem delta_v otimizam a estabilidade
            dists_to_avg = torch.norm(pixels - avg.unsqueeze(1) + 1e-6, dim=0)
            var = torch.mean(torch.clamp(dists_to_avg - delta_v, min=0.0) ** 2)
            
            var_loss += var
            reg_loss += torch.norm(avg)
            centroids.append(avg)

        num_centroids = len(centroids)
        if num_centroids == 0:
            continue

        if num_centroids > 1:
            centroids_tensor = torch.stack(centroids)
            dists = torch.cdist(centroids_tensor, centroids_tensor, p=2.0)
            triu_idx = torch.triu_indices(num_centroids, num_centroids, offset=1)
            pairwise_dists = dists[triu_idx[0], triu_idx[1]]
            
            # Uso de .mean() em vez de .sum() para evitar que dist_loss domine a loss total 
            # quando há muitos objetos na mesma imagem
            dist_loss = torch.mean(torch.clamp(delta_d - pairwise_dists, min=0.0) ** 2)

        # var_loss e reg_loss são normalizados por instância. 
        # dist_loss já foi normalizado corretamente pelo .mean() acima
        loss_emb = (var_loss + reg_loss) / num_centroids + dist_loss
        total_loss += loss_emb

    return total_loss / batch_size


def ablation(
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    device: torch.device,
    axis: int,
    seeds: Sequence[int] = (42, 100),
) -> Dict[str, Tuple[float, float]]:
    if axis == 1:
        architectures = {
            "SegNet": SegNetDDimensional,
            "UNet ": UNetDDimensional,
            "DeepLab": DeepLabDDimensional,
        }
    elif axis == 3:
        architectures = {
            "ParseNet": ParseNetDDimensional,
            "PSPNet": PSPNetDDimensional,
        }
    else:
        raise ValueError("axis deve ser 1 ou 3")

    maP_result_comb: Dict[str, Tuple[float, float]] = {}

    for name, model_class in architectures.items():
        mAP_results: List[float] = []
        for current_seed in seeds:
            print(f"Avaliando Arquitetura: {name} com seed {current_seed}")
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            model = model_class(D=2).to(device)
            train_model(model, dataloader_train, device, part=2, num_epochs=10)
            all_mAPs = evaluate(model, dataloader_val, device, part=2)[0]
            mAP_results.append(np.mean(all_mAPs))
        mean_map = np.mean(mAP_results)
        std_map = np.std(mAP_results)
        maP_result_comb[name] = (mean_map, std_map)

        print(f"\n[Eixo {axis}] Resultado Final: mAP = {mean_map:.4f} ± {std_map:.4f}")
    return maP_result_comb

   