import numpy as np
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from .metrics import calcular_iou
from scipy.optimize import linear_sum_assignment

SEED = 42

def inicializar_objetos(num_objetos, velocidade_tipica,dur_oclusao,  img_size=128, seed=42):
    rng = np.random.default_rng(seed)
    objetos = []
    # O tamanho do raio médio é ajustado para satisfazer a duração desejada da oclusão
    raio_base = max(6, int((dur_oclusao * velocidade_tipica) / 2))
    
    for i in range(num_objetos):
        # Direção aleatória multiplicada pela velocidade típica
        angulo_mov = rng.uniform(0, 2 * np.pi)
        vx = velocidade_tipica * np.cos(angulo_mov)
        vy = velocidade_tipica * np.sin(angulo_mov)

        # Variamos o raio em torno do raio_base calculado
        rx = int(rng.uniform(0.8, 1.2) * raio_base)
        ry = int(rng.uniform(0.8, 1.2) * raio_base)
        
        obj = {
            'id': i + 1,  # ID único do objeto
            'center': rng.integers(20, img_size - 20, size=2).astype(float),
            'vx': vx,
            'vy': vy,
            'axes': (rx, ry),
            'angle': int(rng.integers(0, 180)),
            'color': int(rng.integers(80, 255)),
            'z': rng.random()  # Ordem de profundidade (para oclusão)
        }
        objetos.append(obj)
        
    return objetos


def gerador(num_objetos=8,
            velocidade_tipica=2.0,
            dur_oclusao=10,
            num_frames=45, 
            img_size=(128, 128),
            seed=42):

    w = img_size[0] if isinstance(img_size, tuple) else img_size
    h = img_size[1] if isinstance(img_size, tuple) else img_size

    objetos = inicializar_objetos(
        num_objetos=num_objetos, 
        velocidade_tipica=velocidade_tipica,
        dur_oclusao=dur_oclusao,
        img_size=w,
        seed=seed
    )
    objetos.sort(key=lambda x: x['z'])
    rng = np.random.default_rng(seed)

    video_frames = []
    ground_truth = []

    # LOOP 1: Percorre todos os quadros
    for t in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # LOOP 2: Desenha cada objeto no quadro t
        for obj in objetos:
            cx, cy = obj['center']
            rx, ry = obj['axes']
            centro_int = (int(cx), int(cy))
            cor_rgb = (obj['color'], obj['color'], obj['color'])
          
            cv.ellipse(frame, centro_int, (rx, ry), obj['angle'], 0, 360, cor_rgb, -1)

            bb_left = cx - rx
            bb_top = cy - ry
            bb_width = 2 * rx
            bb_height = 2 * ry

            ground_truth.append([t + 1, obj['id'], bb_left, bb_top, bb_width, bb_height, 1.0, 1, 1.0])
        
            obj['center'][0] += obj['vx']
            obj['center'][1] += obj['vy']

        # FORA DO LOOP 2 (dos objetos), DENTRO DO LOOP 1 (dos quadros)
        contrast = rng.uniform(0.7, 1.3)
        noise = rng.integers(-15, 15, size=(h, w, 3))
        frame = np.clip(frame.astype(float) * contrast + noise, 0, 255).astype(np.uint8)

        video_frames.append(frame)

    # FORA DO LOOP 1 (de quadros): Retorna todos os 45 quadros acumulados
    return video_frames, ground_truth

class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost=3):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 1
        self.tracks = {}  # {track_id: {'box': [x, y, w, h], 'lost': count}}

    def update(self, frame_idx, detections):
        """
        detections: lista de caixas no formato [x, y, w, h, conf]
        retorna: lista no formato MOT [frame_idx, track_id, x, y, w, h, conf]
        """
        results = []
        det_boxes = [d[:4] for d in detections]
        
        # Se não há pistas ativas, inicializa todas as detecções como novas pistas
        if len(self.tracks) == 0:
            for d in detections:
                self.tracks[self.next_id] = {'box': d[:4], 'lost': 0}
                results.append([frame_idx, self.next_id, *d[:4], d[4]])
                self.next_id += 1
            return results

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]['box'] for tid in track_ids]

        if len(det_boxes) == 0:
            # Incrementa o contador de perdidos para todas as pistas
            to_delete = []
            for tid in track_ids:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return results

        # Matriz de IoU entre pistas existentes e novas detecções
        iou_matrix = np.zeros((len(track_ids), len(det_boxes)))
        for i, t_box in enumerate(track_boxes):
            for j, d_box in enumerate(det_boxes):
                iou_matrix[i, j] = calcular_iou(t_box, d_box)

        # Associação via Algoritmo Húngaro
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                tid = track_ids[r]
                d = detections[c]
                
                # Atualiza a pista
                self.tracks[tid]['box'] = d[:4]
                self.tracks[tid]['lost'] = 0
                results.append([frame_idx, tid, *d[:4], d[4]])
                
                matched_tracks.add(r)
                matched_dets.add(c)

        # Trata pistas não associadas
        to_delete = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost:
                    to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # Cria novas pistas para detecções não associadas
        for j, d in enumerate(detections):
            if j not in matched_dets:
                self.tracks[self.next_id] = {'box': d[:4], 'lost': 0}
                results.append([frame_idx, self.next_id, *d[:4], d[4]])
                self.next_id += 1

        return results

