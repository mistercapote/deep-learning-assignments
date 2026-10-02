import numpy as np 
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from .metrics import calcular_iou

#  Part 1.1
def carregar_deteccoes_mot17(caminho_det_txt, min_conf=0.0):
    """
    Carrega detecções públicas de uma sequência MOT17 (ex: det/det.txt).
    
    Format retorno: lista de [frame, -1, x, y, w, h, conf]
    """
    deteccoes = []
    data = np.loadtxt(caminho_det_txt, delimiter=',')
    
    for row in data:
        frame, _, x, y, w, h, conf = row[:7]
        if conf >= min_conf:
            deteccoes.append([int(frame), -1, float(x), float(y), float(w), float(h), float(conf)])
            
    return deteccoes

def carregar_modelo_torchvision(device='cpu'):
    # Carrega modelo pré-treinado em COCO
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model.to(device)

def detectar_quadro_torchvision(model, image_path, frame_idx, min_conf=0.5, device='cpu'):
    """
    Roda inferência em um único quadro de imagem e extrai caixas da classe 'person' (label == 1).
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    deteccoes_frame = []
    for box, score, label in zip(boxes, scores, labels):
        # Filtra apenas a classe 'person' (label == 1) e confiança mínima
        if label == 1 and score >= min_conf:
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            deteccoes_frame.append([int(frame_idx), -1, float(x1), float(y1), float(w), float(h), float(score)])
            
    return deteccoes_frame

def simular_detector(gt, p_drop=0.1, noise_std=2.0, fp_per_frame=0.5, img_size=128):
    """
    Simula um detector imperfeito corrompendo o Ground Truth (GT).
    
    Parâmetros:
    - gt: Lista de bounding boxes verdadeiras no formato [frame_id, obj_id, x, y, w, h]
    - p_drop: Probabilidade (0 a 1) de uma detecção verdadeira ser descartada (falso negativo).
    - noise_std: Desvio padrão do ruído gaussiano adicionado às coordenadas [x, y, w, h].
    - fp_per_frame: Média de falsos positivos injetados por quadro.
    - img_size: Tamanho do vídeo gerado (por padrão 128x128 segundo o enunciado) para limitar FPs.
    
    Retorna:
    - deteccoes: Lista de detecções no formato [frame_id, obj_id, x, y, w, h, conf]
    """
    deteccoes = []
    frames_ids = np.unique([d[0] for d in gt]).astype(int)
    
    # 1. Aplicar drop e ruído nas detecções verdadeiras
    for bbox in gt:
        # Descarta p% das caixas
        if np.random.rand() < p_drop:
            continue
            
        frame_id = bbox[0]
        obj_id = -1 # Trocamos para -1 pois o rastreador ainda não sabe a identidade
        
        # Extrai coordenadas e adiciona ruído gaussiano
        x, y, w, h = bbox[2:6]
        x += np.random.normal(0, noise_std)
        y += np.random.normal(0, noise_std)
        w += np.random.normal(0, noise_std)
        h += np.random.normal(0, noise_std)
        
        # Garante que largura e altura não fiquem negativas ou nulas após o ruído
        w = max(1.0, w)
        h = max(1.0, h)
        
        # Confiança alta simulada para verdadeiros positivos
        conf = np.random.uniform(0.7, 1.0)
        
        deteccoes.append([frame_id, obj_id, x, y, w, h, conf])
        
    # 2. Injetar falsos positivos
    for f in frames_ids:
        # Usamos uma distribuição de Poisson para o número de falsos positivos no frame
        num_fps = np.random.poisson(fp_per_frame)
        
        for _ in range(num_fps):
            # Gera coordenadas aleatórias dentro dos limites da imagem (128x128)
            x = np.random.uniform(0, img_size - 10)
            y = np.random.uniform(0, img_size - 10)
            w = np.random.uniform(5, 25)
            h = np.random.uniform(5, 25)
            
            # Confiança geralmente menor para falsos positivos
            conf = np.random.uniform(0.1, 0.5)
            
            deteccoes.append([f, -1, x, y, w, h, conf])
            
    # Ordena as detecções por frame para garantir o processamento sequencial correto
    deteccoes.sort(key=lambda item: item[0])
    
    return deteccoes

# Parte 1.2

class NaiveIoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost=3):
        """
        Rastreador de Baseline por IoU (Parte 1 - Item 2)
        
        Parameters:
        -----------
        iou_threshold : float
            Limiar mínimo de IoU para aceitar uma associação.
        max_lost : int (k)
            Número máximo de quadros consecutivos sem observação até declarar morte da track.
        """
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.active_tracks = {}  # {track_id: {'box': [x, y, w, h], 'lost_count': int}}

    def update(self, frame_idx, detections):
        """
        Processa as detecções do quadro atual (frame_idx).
        
        detections: lista/array de caixas no formato [x, y, w, h, conf]
        Retorna: lista de predições no formato MOT [frame_idx, track_id, x, y, w, h, conf]
        """
        results = []
        det_boxes = [d[:4] for d in detections]
        
        # 1. Se não há pistas ativas, todas as detecções viram novas pistas
        if len(self.active_tracks) == 0:
            for d in detections:
                self.active_tracks[self.next_id] = {'box': d[:4], 'lost_count': 0}
                results.append([frame_idx, self.next_id, *d[:4], d[4]])
                self.next_id += 1
            return results

        track_ids = list(self.active_tracks.keys())
        track_boxes = [self.active_tracks[tid]['box'] for tid in track_ids]

        # 2. Se não vieram detecções no quadro t, incrementa contadores de perda
        if len(det_boxes) == 0:
            tracks_to_remove = []
            for tid in track_ids:
                self.active_tracks[tid]['lost_count'] += 1
                if self.active_tracks[tid]['lost_count'] > self.max_lost:
                    tracks_to_remove.append(tid)
            for tid in tracks_to_remove:
                del self.active_tracks[tid]
            return results

        # 3. Monta a Matriz de Custos/IoU entre Tracks (t-1) e Detecções (t)
        iou_matrix = np.zeros((len(track_ids), len(det_boxes)))
        for i, t_box in enumerate(track_boxes):
            for j, d_box in enumerate(det_boxes):
                iou_matrix[i, j] = calcular_iou(t_box, d_box)

        # 4. Matching via Algoritmo Húngaro (maximizando IoU)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            # Filtra pelo limiar fixo
            if iou_matrix[r, c] >= self.iou_threshold:
                tid = track_ids[r]
                det = detections[c]
                
                # Atualiza estado da pista associada
                self.active_tracks[tid]['box'] = det[:4]
                self.active_tracks[tid]['lost_count'] = 0
                results.append([frame_idx, tid, *det[:4], det[4]])
                
                matched_tracks.add(r)
                matched_dets.add(c)

        # 5. Gestão de Morte (Tracks sem observação no quadro atual)
        tracks_to_remove = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.active_tracks[tid]['lost_count'] += 1
                if self.active_tracks[tid]['lost_count'] > self.max_lost:
                    tracks_to_remove.append(tid)
        for tid in tracks_to_remove:
            del self.active_tracks[tid]

        # 6. Gestão de Nascimento (Detecções não associadas ganham novo ID)
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.active_tracks[self.next_id] = {'box': det[:4], 'lost_count': 0}
                results.append([frame_idx, self.next_id, *det[:4], det[4]])
                self.next_id += 1

        return results

# Parte 2

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

class RNNTracker:
    def __init__(self, modelo_rnn, iou_threshold=0.3, max_lost=3, device='cpu'):
        """
        Rastreador Baseado em Memória Temporal (Parte 2 - Trilha A)
        """
        self.modelo = modelo_rnn
        self.modelo.eval() # Garante que está em modo de inferência
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.device = device
        
        self.next_id = 1
        # Estrutura: {track_id: {'box': [x,y,w,h], 'h': tensor_hidden, 'lost_count': int}}
        self.active_tracks = {} 
        
    def update(self, frame_idx, detections):
        results = []
        det_boxes = [d[:4] for d in detections]
        
        # Se não há pistas ativas, inicializa todas as detecções como novas
        if len(self.active_tracks) == 0:
            for d in detections:
                # O estado oculto 'h' começa a None (o PyTorch assume zeros)
                self.active_tracks[self.next_id] = {'box': d[:4], 'h': None, 'lost_count': 0}
                results.append([frame_idx, self.next_id, *d[:4], d[4]])
                self.next_id += 1
            return results
            
        track_ids = list(self.active_tracks.keys())
        predicted_boxes = []
        novos_h = []
        
        # 1. PREVISÃO: Rodar a RNN para prever a próxima posição de cada pista
        with torch.no_grad():
            for tid in track_ids:
                track = self.active_tracks[tid]
                # Prepara a entrada: (batch=1, seq_len=1, features=4)
                x_input = torch.tensor(track['box'], dtype=torch.float32).view(1, 1, 4).to(self.device)
                x_input = torch.tensor(track['box'], dtype=torch.float32).view(1, 1, 4).to(self.device)
                h_input = track['h']
                
                # Roda o modelo temporal
                pred_box, h_new = self.modelo(x_input, h_input)
                
                predicted_boxes.append(pred_box.cpu().numpy().squeeze().tolist())
                
                # ---> CORREÇÃO AQUI: Lidando com LSTM (Tupla) e RNN/GRU (Tensor) <---
                # O detach() desvincula o hidden state do grafo computacional anterior,
                # o que é obrigatório para não estourar a memória durante o rastreamento em sequência.
                if isinstance(h_new, tuple):
                    # Se for LSTM, dá detach no hidden state e cell state separadamente
                    h_detached = tuple(h.detach() for h in h_new)
                else:
                    # Se for RNN ou GRU, dá detach direto
                    h_detached = h_new.detach()
                    
                novos_h.append(h_detached)
        
        # Se não há detecções no quadro atual, todas as pistas sofrem oclusão
        if len(det_boxes) == 0:
            tracks_to_remove = []
            for i, tid in enumerate(track_ids):
                self.active_tracks[tid]['box'] = predicted_boxes[i] # Roda para a frente!
                self.active_tracks[tid]['h'] = novos_h[i]
                self.active_tracks[tid]['lost_count'] += 1
                
                if self.active_tracks[tid]['lost_count'] > self.max_lost:
                    tracks_to_remove.append(tid)
                    
            for tid in tracks_to_remove:
                del self.active_tracks[tid]
            return results

        # 2. ASSOCIAÇÃO: Matriz de IoU usando as PREVISÕES (e não a posição antiga)
        iou_matrix = np.zeros((len(track_ids), len(det_boxes)))
        for i, p_box in enumerate(predicted_boxes):
            for j, d_box in enumerate(det_boxes):
                iou_matrix[i, j] = calcular_iou(p_box, d_box)
                
        # Matching via Algoritmo Húngaro
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_tracks = set()
        matched_dets = set()
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                tid = track_ids[r]
                det = detections[c]
                
                # SUCESSO: Atualiza a pista com a observação REAL (correção)
                self.active_tracks[tid]['box'] = det[:4]
                self.active_tracks[tid]['h'] = novos_h[r]
                self.active_tracks[tid]['lost_count'] = 0
                
                results.append([frame_idx, tid, *det[:4], det[4]])
                matched_tracks.add(r)
                matched_dets.add(c)
                
        # 3. GESTÃO DE OCLUSÃO: Tracks não associadas "rodam para a frente"
        tracks_to_remove = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                # Usa a PREVISÃO da rede como a nova posição (fantasma)
                self.active_tracks[tid]['box'] = predicted_boxes[i]
                self.active_tracks[tid]['h'] = novos_h[i]
                self.active_tracks[tid]['lost_count'] += 1
                
                if self.active_tracks[tid]['lost_count'] > self.max_lost:
                    tracks_to_remove.append(tid)
                    
        for tid in tracks_to_remove:
            del self.active_tracks[tid]
            
        # 4. NASCIMENTO: Novas detecções viram novas tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.active_tracks[self.next_id] = {'box': det[:4], 'h': None, 'lost_count': 0}
                results.append([frame_idx, self.next_id, *det[:4], det[4]])
                self.next_id += 1
                
        return results