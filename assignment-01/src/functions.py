import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from sklearn.cluster import DBSCAN
from .models import   ParseNetDDimensional, PSPNetDDimensional, UNetTernary

def calculate_instance_metrics(true_instances, pred_instances):
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


def embeddings_to_instances(pred_bin, pred_emb, eps =1.0, min_samples = 5):
    mask = (torch.sigmoid(pred_bin[0])>0.5).cpu().numpy()
    pred_instances = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return pred_instances

    emb_foreground = (pred_emb[:, mask].T).cpu().numpy()
    
    model = DBSCAN(eps= eps, min_samples= min_samples)
    labels = model.fit_predict(emb_foreground) + 1
    pred_instances[mask]= labels

    return pred_instances


def train_model(model, dataloader, device, part: int, lr: float = 1e-3, num_epochs: int = 5):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_batchs = len(dataloader)

    for epoch in range(num_epochs):
        model.train()
        accumulated_loss = 0.0
        total_intersection = 0.0
        total_union = 0.0
        
        for images, masks, instance_gt in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            if part == 1:
                optimizer.zero_grad()
                prediction_bin = model(images)
                loss = criterion(prediction_bin, masks)
            elif part == 2:
                instance_gt = instance_gt.to(device)
                optimizer.zero_grad()
                prediction_bin, prediction_emb = model(images)
                loss_bin = criterion(prediction_bin, masks)
                loss_emb = discrimative_loss(prediction_emb, instance_gt)
                loss = loss_bin + loss_emb
            loss.backward()
            optimizer.step()

            binary_prediction = (torch.sigmoid(prediction_bin) > 0.5).float()
            intersection = (binary_prediction * masks).sum()
            union = binary_prediction.sum() + masks.sum() - intersection
            
            accumulated_loss += loss.item()
            total_intersection += intersection.item()
            total_union += union.item()
            
        epoch_iou = total_intersection / (total_union + 1e-6)
        epoch_dice = (2.0 * total_intersection) / (total_union + total_intersection + 1e-6)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {accumulated_loss/num_batchs:.4f} | IoU: {epoch_iou:.4f} | Dice: {epoch_dice:.4f}")


def evaluate(model, dataloader, device, part: int, k_samples: int = 4):
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
                pred_instances
            )
            
            all_mAPs.append(mAP)
            all_count_errors.append(count_error)
            all_densities.append(density)
            
            # Transpor imagem (C, H, W) para (H, W, C) para o matplotlib
            img_plot = images[i].cpu().numpy().transpose(1, 2, 0)
            samples.append((count_error, img_plot, real_instances_np[i], pred_instances))
            
    if len(samples) > k_samples:
        index = np.random.choice(len(samples), size=k_samples, replace=False)
        samples = [samples[i] for i in index]
    
    return all_mAPs, all_count_errors, all_densities, samples


def discrimative_loss(prediction, instance, delta_d=1.5):
    batch_size = prediction.size(0)
    total_loss = prediction.sum() * 0.0 
    
    for b in range(batch_size):
        pred_b = prediction[b] # [C, H, W]
        inst_b = instance[b]   # [H, W]
        objects_ids = torch.unique(inst_b)[1:] 
        var_loss = 0.0
        dist_loss = 0.0
        reg_loss = 0.0
        centroids = []

        for obj_id in objects_ids:
            mask = (inst_b == obj_id)
            pixels = pred_b[:, mask] 
            
            avg = torch.mean(pixels, dim=1)
            var = torch.mean(torch.norm(pixels - avg.unsqueeze(1), dim=0))
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
            dist_loss = torch.clamp(delta_d - pairwise_dists, min=0).sum()
        loss_emb = (reg_loss + dist_loss + var_loss) / num_centroids
        total_loss += loss_emb
        
    return total_loss / batch_size


def ablation(dataloader_train, dataloader_val, device, axis, seeds: list[int] = [42, 100]):
    
    architectures = {
         
                  "UNeT": UNetTernary
        }

    maP_result_comb = {}
   
    for name, model_class in architectures.items():
        mAP_results = []
        for current_seed in seeds:
            print(f"Avaliando Arquitetura: {name} com seed {current_seed}")
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            model = model_class().to(device)
            train_model(model, dataloader_train, device, part=2, num_epochs=10)
            all_mAPs = evaluate(model, dataloader_val, device, part=2)[0]
            mAP_results.append(np.mean(all_mAPs))
        mean_map = np.mean(mAP_results)
        std_map = np.std(mAP_results)
        maP_result_comb[name] = (mean_map,std_map )
    

        print(f"\n[Eixo {axis}] Resultado Final: mAP = {mean_map:.4f} ± {std_map:.4f}")
    return maP_result_comb

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
