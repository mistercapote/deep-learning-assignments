import numpy as np
from .metrics import * 

def calcular_map_quadro(gt_boxes, pred_boxes, iou_thresholds=np.arange(0.50, 1.00, 0.05)):
    """
    Calcula o Average Precision (AP) de um ÚNICO QUADRO utilizando a abordagem 
    gulosa do PA1 (IoU decrescente) adaptada para Bounding Boxes.
    
    gt_boxes, pred_boxes: listas de formato [x, y, w, h] (sem frame ou ID)
    """
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    # Se não há GT nem predições, o acerto é 100%
    if n_gt == 0 and n_pred == 0:
        return 1.0
        
    # Se um dos dois é vazio e o outro não, a precisão do quadro é 0
    if n_gt == 0 or n_pred == 0:
        return 0.0

    # 1. Montar Matriz de IoU usando geometria [x, y, w, h]
    iou_matrix = np.zeros((n_pred, n_gt))
    for i, p_box in enumerate(pred_boxes):
        for j, g_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calcular_iou(p_box, g_box) # Reutiliza a função que já criámos

    # 2. Pareamento Guloso (Idêntico ao PA1)
    pairs = [
        (iou_matrix[i, j], i, j)
        for i in range(n_pred)
        for j in range(n_gt)
        if iou_matrix[i, j] > 0
    ]
    pairs.sort(key=lambda x: -x[0]) # Ordenar por IoU decrescente

    img_aps = np.zeros(len(iou_thresholds))

    # 3. Calcular TP, FP, FN para cada limiar
    for k, t in enumerate(iou_thresholds):
        matched_pred = set()
        matched_gt = set()
        tp = 0
        
        for iou, i, j in pairs:
            if iou < t:
                break # Como está ordenado, os próximos serão ainda menores
            if i not in matched_pred and j not in matched_gt:
                matched_pred.add(i)
                matched_gt.add(j)
                tp += 1
                
        fp = n_pred - tp
        fn = n_gt - tp
        denom = tp + fp + fn
        
        img_aps[k] = tp / denom if denom > 0 else 1.0

    # Retorna o mAP (Média dos APs para os limiares 0.5 a 0.95)
    return float(img_aps.mean())