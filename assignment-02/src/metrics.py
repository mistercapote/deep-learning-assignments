import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def calcular_iou(b1, b2):
    """Calcula IoU entre duas bboxes no formato [x, y, w, h]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

# O idf1 mede a qualidade global da identificação (em %)
def calcular_idf1(gt, predicoes, iou_threshold=0.5):
    """
    Calcula o IDF1 com atribuição global 1-para-1 entre IDs de GT e Predições.
    gt e predicoes: listas no formato [frame, id, x, y, w, h, ...]
    """
    if len(gt) == 0 or len(predicoes) == 0:
        return 0.0

    gt_arr = np.array(gt)
    pred_arr = np.array(predicoes)

    gt_ids = np.unique(gt_arr[:, 1]).astype(int)
    pred_ids = np.unique(pred_arr[:, 1]).astype(int)

    gt_id_to_idx = {gid: i for i, gid in enumerate(gt_ids)}
    pred_id_to_idx = {pid: j for j, pid in enumerate(pred_ids)}

    # Matriz de afinidade temporal: contagem de correspondências com IoU >= threshold
    cost_matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=int)

    # Agrupa bboxes por frame para comparar
    frames = np.unique(gt_arr[:, 0]).astype(int)
    for f in frames:
        gt_f = [d for d in gt if d[0] == f]
        pr_f = [d for d in predicoes if d[0] == f]

        for g in gt_f:
            g_idx = gt_id_to_idx[int(g[1])]
            for p in pr_f:
                p_idx = pred_id_to_idx[int(p[1])]
                if calcular_iou(g[2:6], p[2:6]) >= iou_threshold:
                    cost_matrix[g_idx, p_idx] += 1

    # Atribuição global ótima maximizando os matches (minimizando -cost)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    idtp = cost_matrix[row_ind, col_ind].sum()

    n_gt = len(gt)
    n_pred = len(predicoes)

    idfp = n_pred - idtp
    idfn = n_gt - idtp

    denominador = 2 * idtp + idfp + idfn
    return float(2 * idtp / denominador) if denominador > 0 else 0.0


#  IDSW mede a frequência de erros de troca de identidade (em contagem absoluta).
def calcular_idsw_e_fragmentacoes(gt, predicoes, iou_threshold=0.5):
    """
    Calcula ID Switches (IDSW) e Fragmentações (Frag) frame a frame.
    - IDSW: Um objeto do GT troca de identificador predito associado.
    - Frag: A trajetória do GT é interrompida (perde rastreio) e retomada posteriormente.
    """
    if len(gt) == 0 or len(predicoes) == 0:
        return 0, 0

    gt_frames = np.unique([d[0] for d in gt]).astype(int)
    pr_frames = np.unique([d[0] for d in predicoes]).astype(int)
    todos_frames = sorted(list(set(gt_frames).union(set(pr_frames))))

    # Estado de cada GT: último ID predito associado e último frame rastreado
    ultimo_pred_id = {}
    ultimo_frame_rastreado = {}

    idsw = 0
    fragmentacoes = 0

    for f in todos_frames:
        gt_f = [d for d in gt if d[0] == f]
        pr_f = [d for d in predicoes if d[0] == f]

        # Matching frame a frame para descobrir qual predição pegou qual GT neste instante
        if len(gt_f) > 0 and len(pr_f) > 0:
            iou_mat = np.zeros((len(gt_f), len(pr_f)))
            for i, g in enumerate(gt_f):
                for j, p in enumerate(pr_f):
                    iou_mat[i, j] = calcular_iou(g[2:6], p[2:6])

            r_ind, c_ind = linear_sum_assignment(-iou_mat)
            matches_frame = {}
            for r, c in zip(r_ind, c_ind):
                if iou_mat[r, c] >= iou_threshold:
                    matches_frame[int(gt_f[r][1])] = int(pr_f[c][1])
        else:
            matches_frame = {}

        # Avalia switches e fragmentações para cada objeto de GT presente no frame
        for g in gt_f:
            gid = int(g[1])
            if gid in matches_frame:
                curr_pid = matches_frame[gid]

                if gid in ultimo_pred_id:
                    # 1. ID Switch: se trocou o ID predito associado
                    if curr_pid != ultimo_pred_id[gid]:
                        idsw += 1
                    
                    # 2. Fragmentação: se houve um salto de frames sem rastreamento
                    if f > ultimo_frame_rastreado[gid] + 1:
                        fragmentacoes += 1

                ultimo_pred_id[gid] = curr_pid
                ultimo_frame_rastreado[gid] = f

    return idsw, fragmentacoes


def calcular_metricas_completas(gt, predicoes, iou_threshold=0.5):
    """
    Agrupa todas as métricas obrigatórias da Parte 1.3.
    """
    gt_arr = np.array(gt)
    pr_arr = np.array(predicoes)

    ids_true = len(np.unique(gt_arr[:, 1])) if len(gt_arr) > 0 else 0
    ids_pred = len(np.unique(pr_arr[:, 1])) if len(pr_arr) > 0 else 0

    idf1 = calcular_idf1(gt, predicoes, iou_threshold=iou_threshold)
    idsw, frag = calcular_idsw_e_fragmentacoes(gt, predicoes, iou_threshold=iou_threshold)

    # Análogo temporal do erro de contagem do PA1
    erro_absoluto_contagem = abs(ids_pred - ids_true)
    razao_sobrecontagem = (ids_pred / ids_true) if ids_true > 0 else 0.0

    return {
        'idf1': idf1,
        'idsw': idsw,
        'frag': frag,
        'ids_true': ids_true,
        'ids_pred': ids_pred,
        'erro_contagem': erro_absoluto_contagem,
        'razao_ids': razao_sobrecontagem,
        'idsw_por_gt': (idsw / ids_true) if ids_true > 0 else 0.0
    }


def analisar_sobrevivencia_oclusao(gt, predicoes):
    """
    Analisa os buracos (gaps/oclusões) no Ground Truth e verifica se o
    ID previsto sobreviveu a oclusão. (Parte 4.2)
    """
    gt_arr = np.array(gt)
    pr_arr = np.array(predicoes)
    
    if len(gt_arr) == 0 or len(pr_arr) == 0:
        return
        
    gt_ids = np.unique(gt_arr[:, 1]).astype(int)
    
    duracao_oclusoes = []
    sobrevivencias = []
    
    # Dicionário rápido para achar predições: {(frame, gt_id): pred_id}
    # (Para simplificar, vamos assumir que fizemos um match de IoU rápido aqui)
    # Recomendado usar sua função de matches_frame da métrica original.
    
    for gid in gt_ids:
        # Pega os frames onde este ID verdadeiro aparece
        frames_gt = sorted(gt_arr[gt_arr[:, 1] == gid][:, 0].astype(int))
        
        # Procura por saltos/buracos nos frames (oclusão)
        for i in range(1, len(frames_gt)):
            f_prev = frames_gt[i-1]
            f_curr = frames_gt[i]
            gap = f_curr - f_prev - 1
            
            if gap > 0:
                # É uma oclusão!
                duracao_oclusoes.append(gap)
                
                # Pegar o pr_id no frame_prev e frame_curr (lógica simplificada assumindo match exato de bounding box, na prática use IoU > 0.5)
                box_prev = gt_arr[(gt_arr[:, 0] == f_prev) & (gt_arr[:, 1] == gid)][0][2:6]
                box_curr = gt_arr[(gt_arr[:, 0] == f_curr) & (gt_arr[:, 1] == gid)][0][2:6]
                
                pr_prev = [p for p in predicoes if p[0] == f_prev and calcular_iou(p[2:6], box_prev) > 0.3]
                pr_curr = [p for p in predicoes if p[0] == f_curr and calcular_iou(p[2:6], box_curr) > 0.3]
                
                if len(pr_prev) > 0 and len(pr_curr) > 0:
                    id_antes = pr_prev[0][1]
                    id_depois = pr_curr[0][1]
                    sobrevivencias.append(1 if id_antes == id_depois else 0)
                else:
                    sobrevivencias.append(0) # Perdeu o rastro
                    
    # Plotar o histograma de oclusões x taxa de sobrevivência
    if len(duracao_oclusoes) > 0:
        plt.figure(figsize=(8, 4))
        plt.scatter(duracao_oclusoes, sobrevivencias, alpha=0.5)
        plt.xlabel("Duração da Oclusão (Quadros)")
        plt.ylabel("Sobreviveu (1=Sim, 0=Não)")
        plt.title("Horizonte Empírico de Memória")
        plt.show()
    else:
        print("Nenhuma oclusão encontrada no Ground Truth fornecido.")