import numpy as np
from scipy.optimize import linear_sum_assignment



def calcular_iou(box1, box2):
    """
    O formato de cada caixa é (x, y, w, h)
    """

    xA, yA, wA, hA = box1
    xB, yB, wB, hB = box2

    left = max(xA, xB)
    top = max(yA, yB)

    right = min(xA + wA, xB + wB)
    bottom = min(yA + hA, yB + hB)

    largura_int = max(0.0, right - left)
    altura_int = max(0.0, bottom - top)

    area_inter = largura_int * altura_int

    if area_inter == 0:
        return 0.0

    areabox1 = wA * hA
    areabox2 = wB * hB

    area_uniao = areabox1 + areabox2 - area_inter

    return area_inter / area_uniao

# O idf1 mede a qualidade global da identificação (em %)
def calcular_idf1 (gt_list, tr_list, iou_thre = 0.5):


    if len(gt_list) ==0 or len(tr_list)==0:
        return 0.0


    gt_arr = np.array(gt_list)
    tr_arr = np.array(tr_list)

    gt_ids = np.unique(gt_arr[:, 1]).astype(int)
    tr_ids = np.unique(tr_arr[:, 1]).astype(int)

    num_gt_ids = len(gt_ids)
    num_tr_ids = len(tr_ids)

    gt_id_to_idx = {g_id: idx for idx, g_id in enumerate(gt_ids)}
    tr_id_to_idx = {t_id: idx for idx, t_id in enumerate(tr_ids)}
    #  matriz de acertos
    cost_matrix = np.zeros((num_gt_ids, num_tr_ids), dtype= int)

    gt_dict = {(int(f), int(i)): box for f, i, *box in gt_arr[:, :6]}
    tr_dict = {(int(f), int(i)): box for f, i, *box in tr_arr[:, :6]}

    frames = np.unique(np.concatenate([gt_arr[:, 0], tr_arr[:, 0]])).astype(int)

    for f in frames:
        gt_in_frame = [i for i in gt_ids if (f, i) in gt_dict]
        tr_in_frame = [j for j in tr_ids if (f, j) in tr_dict]

        for g_id in gt_in_frame:
            box_gt = gt_dict[(f, g_id)]
            i_idx = gt_id_to_idx[g_id]
            
            for t_id in tr_in_frame:
                box_tr = tr_dict[(f, t_id)]
                j_idx = tr_id_to_idx[t_id]

                if calcular_iou(box_gt, box_tr) >= iou_thre:
                    cost_matrix[i_idx, j_idx] += 1

    #  Algoritmo Húngaro para Associação Global 1-para-1 (Maximização)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    # Total de acertos globais (IDTP)
    idtp = cost_matrix[row_ind, col_ind].sum()

    n_gt = len(gt_list)
    n_tr = len(tr_list)

    # Cálculo final do IDF1
    idf1 = (2.0 * idtp) / (n_gt + n_tr) if (n_gt + n_tr) > 0 else 0.0

    return idf1


#  IDSW mede a frequência de erros de troca de identidade (em contagem absoluta).
def calcular_idsw(gt_list, tr_list, iou_threshold=0.5):
    """
    Calcula a quantidade de ID Switches (IDSW) na sequência.
    
    gt_list, tr_list: listas no formato [frame, id, x, y, w, h, ...]
    """
    if len(gt_list) == 0 or len(tr_list) == 0:
        return 0

    gt_arr = np.array(gt_list)
    tr_arr = np.array(tr_list)

    # Identifica todos os quadros presentes na sequência
    frames = np.unique(np.concatenate([gt_arr[:, 0], tr_arr[:, 0]])).astype(int)
    frames.sort()

    # Dicionário para guardar o último ID do rastreador associado a cada ID do GT
    ultimo_tr_id_do_gt = {}
    idsw_count = 0

    for f in frames:
        # Filtra caixas do frame atual
        gt_frame = gt_arr[gt_arr[:, 0] == f]
        tr_frame = tr_arr[tr_arr[:, 0] == f]

        if len(gt_frame) == 0 or len(tr_frame) == 0:
            continue

        # Matriz de IoU para o frame f
        iou_matrix = np.zeros((len(gt_frame), len(tr_frame)))
        for i, g_row in enumerate(gt_frame):
            for j, t_row in enumerate(tr_frame):
                iou_matrix[i, j] = calcular_iou(g_row[2:6], t_row[2:6])

        # Associação húngara no frame atual (maximizando IoU)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                gt_id = int(gt_frame[r, 1])
                tr_id = int(tr_frame[c, 1])

                # Verifica se o GT já foi visto antes e se o ID do rastreador mudou
                if gt_id in ultimo_tr_id_do_gt:
                    if ultimo_tr_id_do_gt[gt_id] != tr_id:
                        idsw_count += 1

                # Atualiza o último ID associado ao GT
                ultimo_tr_id_do_gt[gt_id] = tr_id

    return idsw_count