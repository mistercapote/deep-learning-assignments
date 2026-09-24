import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

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






def visualizar_trajetoria(video_frames, inicio=0, fim=25, passo=2):
    # Garante que não passamos do tamanho total da lista de quadros
    fim_real = min(fim, len(video_frames))
    quadros_para_mostrar = list(range(inicio, fim_real, passo))
    num_quadros = len(quadros_para_mostrar)

    if num_quadros == 0:
        print("Nenhum quadro válido no intervalo especificado.")
        return

    fig, axes = plt.subplots(1, num_quadros, figsize=(3 * num_quadros, 3))
    
    # Se houver apenas 1 quadro, axes não é um array/lista, então convertemos
    if num_quadros == 1:
        axes = [axes]

    for idx, frame_idx in enumerate(quadros_para_mostrar):
        axes[idx].imshow(video_frames[frame_idx])
        axes[idx].set_title(f"Quadro {frame_idx + 1}")
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.show()