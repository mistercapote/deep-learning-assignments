import numpy as np
import torch
from .evaluating import decode_watershed
import torch.nn.functional as F



def create_mosaic_real(dataset, indices=[0, 1, 2, 3]):
    """
    Cria uma imagem grande (mosaico 2x2) combinando 4 amostras do dataset real.
    """

    def instances_to_label_mask(instances):
        """
        Converte uma lista de máscaras individuais em uma única máscara
        onde cada instância possui um ID diferente.
        """
        # Caso seja tensor
        if hasattr(instances, "detach"):
            instances = instances.cpu().numpy()

        # Caso já seja uma máscara NumPy [H, W]
        if isinstance(instances, np.ndarray) and instances.ndim == 2:
            return instances.astype(np.int32)

        # Caso seja tensor/array [N, H, W]
        if isinstance(instances, np.ndarray) and instances.ndim == 3:
            label_mask = np.zeros(
                instances.shape[1:],
                dtype=np.int32
            )

            for i, mask in enumerate(instances):
                if hasattr(mask, "detach"):
                    mask = mask.cpu().numpy()

                label_mask[mask > 0] = i + 1

            return label_mask

        # Caso seja lista de máscaras
        if isinstance(instances, list):

            # Descobre H e W pela primeira máscara
            first_mask = instances[0]

            if hasattr(first_mask, "detach"):
                first_mask = first_mask.cpu().numpy()

            H, W = first_mask.shape[-2:]

            label_mask = np.zeros((H, W), dtype=np.int32)

            for i, mask in enumerate(instances):

                if hasattr(mask, "detach"):
                    mask = mask.cpu().numpy()

                mask = np.asarray(mask)

                # Cada máscara recebe um ID único
                label_mask[mask > 0] = i + 1

            return label_mask

        raise TypeError(
            f"Formato de instâncias não suportado: "
            f"{type(instances)}"
        )


    # Dataset retorna 5 elementos
    sample_img, _, _, sample_inst, _ = dataset[indices[0]]

    # Converte imagem para NumPy
    if hasattr(sample_img, "detach"):
        sample_img = sample_img.cpu().numpy()

    # Converte lista/stack de instâncias em máscara [H, W]
    sample_inst = instances_to_label_mask(sample_inst)

    C, H, W = sample_img.shape

    # Cria mosaico
    mosaic_img = np.zeros(
        (C, H * 2, W * 2),
        dtype=sample_img.dtype
    )

    mosaic_inst = np.zeros(
        (H * 2, W * 2),
        dtype=np.int32
    )

    positions = [
        (0, 0, indices[0]),
        (0, W, indices[1]),
        (H, 0, indices[2]),
        (H, W, indices[3])
    ]

    max_inst_id = 0

    for y_offset, x_offset, idx in positions:

        # Pega imagem e lista de máscaras individuais
        img, _, _, inst, _ = dataset[idx]

        if hasattr(img, "detach"):
            img = img.cpu().numpy()

        # Converte para máscara com IDs
        inst = instances_to_label_mask(inst)

        # Coloca a imagem no mosaico
        mosaic_img[
            :,
            y_offset:y_offset + H,
            x_offset:x_offset + W
        ] = img

        # Apenas IDs positivos representam objetos
        valid_mask = inst > 0

        # Desloca os IDs para não repetir entre quadrantes
        inst_shifted = inst.copy()

        inst_shifted[valid_mask] += max_inst_id

        if valid_mask.any():
            max_inst_id = inst_shifted.max()

        # Coloca as instâncias no mosaico
        mosaic_inst[
            y_offset:y_offset + H,
            x_offset:x_offset + W
        ] = inst_shifted

    return mosaic_img, mosaic_inst



def mosaic_tile_inference_demo(mosaic_img, tile_size=128, overlap=32):

    C, H, W = mosaic_img.shape

    stride = tile_size - overlap

    tiles = []
    coords = []

    # posições verticais
    y_positions = list(range(0, H - tile_size + 1, stride))

    if y_positions[-1] != H - tile_size:
        y_positions.append(H - tile_size)

    # posições horizontais
    x_positions = list(range(0, W - tile_size + 1, stride))

    if x_positions[-1] != W - tile_size:
        x_positions.append(W - tile_size)

    for y in y_positions:
        for x in x_positions:

            tile = mosaic_img[
                :,
                y:y + tile_size,
                x:x + tile_size
            ]

            tiles.append(tile)

            coords.append(
                (y, y + tile_size, x, x + tile_size)
            )

    return tiles, coords


def mosaic_inference_naive_watershed(
    model,
    mosaic_img,
    tile_size,
    stride,
    device,
    interior_thresh=0.4,
    fg_thresh=0.5,
    min_marker_size=5,
):
    """
    Inferência em mosaico usando o modelo da Trilha A.

    Entrada:
        mosaic_img: imagem no formato (C, H, W)

    Saída:
        lista de máscaras de instâncias no tamanho completo do mosaico.

    Não realiza fusão entre instâncias provenientes de tiles diferentes.
    """

    model.eval()

    # --------------------------------------------------
    # Garantir que mosaic_img seja numpy
    # --------------------------------------------------

    if torch.is_tensor(mosaic_img):
        mosaic_img = mosaic_img.cpu().numpy()

    if mosaic_img.ndim != 3:
        raise ValueError(
            f"Esperado mosaic_img com 3 dimensões (C,H,W), "
            f"mas recebido {mosaic_img.shape}"
        )

    C, H, W = mosaic_img.shape

    if C != 3:
        raise ValueError(
            f"O modelo espera 3 canais, mas mosaic_img possui {C}."
        )

    all_instances = []

    # --------------------------------------------------
    # Percorrer o mosaico em tiles
    # --------------------------------------------------

    with torch.no_grad():

        for y in range(0, H, stride):

            for x in range(0, W, stride):

                # Coordenadas do tile
                y1 = y
                x1 = x
                y2 = min(y + tile_size, H)
                x2 = min(x + tile_size, W)

                tile = mosaic_img[:, y1:y2, x1:x2]

                tile_h = y2 - y1
                tile_w = x2 - x1

                # --------------------------------------------------
                # Padding para tiles nas bordas
                # --------------------------------------------------

                if tile_h < tile_size or tile_w < tile_size:

                    padded_tile = np.zeros(
                        (C, tile_size, tile_size),
                        dtype=mosaic_img.dtype
                    )

                    padded_tile[:, :tile_h, :tile_w] = tile

                    tile = padded_tile

                # --------------------------------------------------
                # Tensor: C,H,W -> 1,C,H,W
                # --------------------------------------------------

                tile_tensor = torch.from_numpy(
                    tile
                ).float().unsqueeze(0).to(device)

                # Normalização
                if tile_tensor.max() > 1:
                    tile_tensor = tile_tensor / 255.0

                # --------------------------------------------------
                # Modelo Trilha A
                # --------------------------------------------------

                logits_cls, dist_pred = model(tile_tensor)

                # Classes:
                # 0 = background
                # 1 = interior
                # 2 = boundary
                probs = F.softmax(logits_cls, dim=1)[0]
                probs = probs.cpu().numpy()

                dist = dist_pred[0, 0].cpu().numpy()

                # --------------------------------------------------
                # Watershed
                # --------------------------------------------------

                tile_instances = decode_watershed(
                    probs,
                    dist,
                    interior_thresh=interior_thresh,
                    fg_thresh=fg_thresh,
                    min_marker_size=min_marker_size,
                )

                # --------------------------------------------------
                # Converter cada instância para coordenadas
                # do mosaico
                # --------------------------------------------------

                for instance_mask in tile_instances:

                    # Retirar padding
                    instance_mask = instance_mask[
                        :tile_h,
                        :tile_w
                    ]

                    if instance_mask.sum() == 0:
                        continue

                    # Máscara no tamanho completo do mosaico
                    full_mask = np.zeros(
                        (H, W),
                        dtype=np.uint8
                    )

                    full_mask[
                        y1:y2,
                        x1:x2
                    ] = instance_mask

                    all_instances.append(full_mask)

                # --------------------------------------------------
                # Evitar tiles redundantes no final da linha
                # --------------------------------------------------

                if x2 == W:
                    break

            # Evitar tiles redundantes no final da imagem
            if y2 == H:
                break

    return all_instances



def mosaic_inference_watershed_with_fusion(
    model,
    large_image,
    tile_size=128,
    overlap=32,
    device='cpu',
    interior_thresh=0.4,
    fg_thresh=0.5,
    min_marker_size=5,
    fusion_iou_thresh=0.3,
):
    """
    Inferência em mosaico para a Trilha A (Watershed).

    Cada tile é processado individualmente e suas instâncias são
    posteriormente fundidas quando apresentam alta sobreposição
    (IoU) na região compartilhada entre os tiles.

    Retorna:
        global_instances: máscara [H, W] com um ID inteiro por instância.
    """

    model.eval()

    # ---------------------------------------------------------
    # Preparação da imagem
    # ---------------------------------------------------------

    if torch.is_tensor(large_image):
        large_image = large_image.detach().cpu().numpy()

    if large_image.ndim != 3:
        raise ValueError(
            f"Esperado large_image no formato (C,H,W), "
            f"mas recebido {large_image.shape}"
        )

    C, H, W = large_image.shape

    if C != 3:
        raise ValueError(
            f"O modelo espera 3 canais, mas recebeu {C}."
        )

    stride = tile_size - overlap

    if stride <= 0:
        raise ValueError(
            "overlap deve ser menor que tile_size."
        )

    # ---------------------------------------------------------
    # Posições dos tiles
    # ---------------------------------------------------------

    y_positions = list(range(0, H - tile_size + 1, stride))

    if not y_positions:
        y_positions = [0]
    elif y_positions[-1] != H - tile_size:
        y_positions.append(H - tile_size)

    x_positions = list(range(0, W - tile_size + 1, stride))

    if not x_positions:
        x_positions = [0]
    elif x_positions[-1] != W - tile_size:
        x_positions.append(W - tile_size)

    # ---------------------------------------------------------
    # Lista de instâncias globais
    # ---------------------------------------------------------

    global_masks = []

    # ---------------------------------------------------------
    # Inferência
    # ---------------------------------------------------------

    with torch.no_grad():

        for y_start in y_positions:

            for x_start in x_positions:

                y_end = min(y_start + tile_size, H)
                x_end = min(x_start + tile_size, W)

                tile = large_image[
                    :,
                    y_start:y_end,
                    x_start:x_end
                ]

                tile_h = y_end - y_start
                tile_w = x_end - x_start

                # -------------------------------------------------
                # Padding caso o tile seja menor na borda
                # -------------------------------------------------

                if tile_h < tile_size or tile_w < tile_size:

                    padded_tile = np.zeros(
                        (C, tile_size, tile_size),
                        dtype=large_image.dtype
                    )

                    padded_tile[
                        :,
                        :tile_h,
                        :tile_w
                    ] = tile

                    tile = padded_tile

                # -------------------------------------------------
                # Tensor
                # -------------------------------------------------

                tile_tensor = (
                    torch.from_numpy(tile)
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )

                # Caso a imagem esteja em [0,255]
                if tile_tensor.max() > 1:
                    tile_tensor = tile_tensor / 255.0

                # -------------------------------------------------
                # Modelo da Trilha A
                # -------------------------------------------------

                logits_cls, dist_pred = model(tile_tensor)

                probs = (
                    F.softmax(logits_cls, dim=1)[0]
                    .cpu()
                    .numpy()
                )

                dist = (
                    dist_pred[0, 0]
                    .cpu()
                    .numpy()
                )

                # -------------------------------------------------
                # Watershed
                # -------------------------------------------------

                tile_instances = decode_watershed(
                    probs,
                    dist,
                    interior_thresh=interior_thresh,
                    fg_thresh=fg_thresh,
                    min_marker_size=min_marker_size,
                )

                # -------------------------------------------------
                # Cada instância encontrada no tile
                # -------------------------------------------------

                for local_mask in tile_instances:

                    # Remove padding
                    local_mask = local_mask[
                        :tile_h,
                        :tile_w
                    ].astype(bool)

                    if local_mask.sum() == 0:
                        continue

                    # -------------------------------------------------
                    # Coloca a máscara no sistema de coordenadas
                    # global do mosaico
                    # -------------------------------------------------

                    global_mask = np.zeros(
                        (H, W),
                        dtype=bool
                    )

                    global_mask[
                        y_start:y_end,
                        x_start:x_end
                    ] = local_mask

                    # -------------------------------------------------
                    # Procurar uma instância existente para fundir
                    # -------------------------------------------------

                    best_match = None
                    best_iou = 0.0

                    for idx, existing_mask in enumerate(global_masks):

                        # Região de interseção espacial entre os
                        # dois tiles
                        overlap_y1 = max(
                            y_start,
                            np.where(existing_mask)[0].min()
                        )

                        overlap_y2 = min(
                            y_end,
                            np.where(existing_mask)[0].max() + 1
                        )

                        overlap_x1 = max(
                            x_start,
                            np.where(existing_mask)[1].min()
                        )

                        overlap_x2 = min(
                            x_end,
                            np.where(existing_mask)[1].max() + 1
                        )

                        # Não há região espacial em comum
                        if (
                            overlap_y1 >= overlap_y2
                            or overlap_x1 >= overlap_x2
                        ):
                            continue

                        # -------------------------------------------------
                        # Máscaras somente na região de overlap
                        # -------------------------------------------------

                        existing_overlap = existing_mask[
                            overlap_y1:overlap_y2,
                            overlap_x1:overlap_x2
                        ]

                        local_overlap = global_mask[
                            overlap_y1:overlap_y2,
                            overlap_x1:overlap_x2
                        ]

                        intersection = np.logical_and(
                            existing_overlap,
                            local_overlap
                        ).sum()

                        union = np.logical_or(
                            existing_overlap,
                            local_overlap
                        ).sum()

                        if union == 0:
                            continue

                        iou = intersection / union

                        if iou > best_iou:
                            best_iou = iou
                            best_match = idx

                    # -------------------------------------------------
                    # Fusão
                    # -------------------------------------------------

                    if (
                        best_match is not None
                        and best_iou >= fusion_iou_thresh
                    ):

                        global_masks[best_match] = np.logical_or(
                            global_masks[best_match],
                            global_mask
                        )

                    else:

                        global_masks.append(global_mask)

    # ---------------------------------------------------------
    # Converter lista de máscaras para label mask
    # ---------------------------------------------------------

    global_instances = np.zeros(
        (H, W),
        dtype=np.int32
    )

    for instance_id, mask in enumerate(global_masks, start=1):

        global_instances[mask] = instance_id

    return global_instances


def evaluate_mosaic_instances(pred_masks, gt_masks):
    """
    Avalia instâncias de um mosaico usando a mesma regra de
    matching e AP definida em evaluating.py.

    pred_masks: lista de máscaras HxW
    gt_masks: lista de máscaras HxW

    Retorna:
        aps
        map_score
        count_error
    """
    IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)

    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    count_error = abs(n_pred - n_gt)

    # Caso não existam instâncias
    if n_pred == 0 and n_gt == 0:
        aps = np.ones(len(IOU_THRESHOLDS))
        return aps, aps.mean(), count_error

    if n_pred == 0 or n_gt == 0:
        aps = np.zeros(len(IOU_THRESHOLDS))
        return aps, aps.mean(), count_error

    # ---------------------------------------------------------
    # Matriz de IoU
    # ---------------------------------------------------------

    pred_flat = np.array(
        pred_masks,
        dtype=np.int32
    ).reshape(n_pred, -1)

    gt_flat = np.array(
        gt_masks,
        dtype=np.int32
    ).reshape(n_gt, -1)

    inter = np.dot(
        pred_flat,
        gt_flat.T
    )

    area_pred = pred_flat.sum(axis=1)[:, np.newaxis]
    area_gt = gt_flat.sum(axis=1)[np.newaxis, :]

    union = area_pred + area_gt - inter

    iou_matrix = inter / np.maximum(
        union,
        1e-6
    )

    # ---------------------------------------------------------
    # Pares ordenados por IoU decrescente
    # ---------------------------------------------------------

    pairs = [
        (iou_matrix[i, j], i, j)
        for i in range(n_pred)
        for j in range(n_gt)
        if iou_matrix[i, j] > 0
    ]

    pairs.sort(
        key=lambda x: -x[0]
    )

    # ---------------------------------------------------------
    # AP para cada threshold
    # ---------------------------------------------------------

    aps = np.zeros(
        len(IOU_THRESHOLDS)
    )

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

        denom = tp + fp + fn

        aps[k] = (
            tp / denom
            if denom > 0
            else 1.0
        )

    map_score = aps.mean()

    return aps, map_score, count_error

def label_mask_to_instances(label_mask):
    instances = []

    for instance_id in np.unique(label_mask):
        if instance_id == 0:
            continue

        mask = (label_mask == instance_id).astype(np.uint8)
        instances.append(mask)

    return instances