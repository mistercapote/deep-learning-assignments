
import matplotlib.pyplot as plt
import numpy as np

#  Parte 0

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


#  Parte 1 

def plotar_descolamento(resultados_sequencias):
    # Ordenar as sequências pelo eixo de dificuldade (Densidade crescente)
    dados_ordenados = sorted(resultados_sequencias, key=lambda x: x['densidade'])
    
    nomes_seq = [d['seq_name'] for d in dados_ordenados]
    map_scores = [d['map'] for d in dados_ordenados]
    idf1_scores = [d['idf1'] for d in dados_ordenados]
    
    razao_ids = [d['ids_pred'] / d['ids_true'] for d in dados_ordenados]
    razao_idsw = [d['idsw'] / d['ids_true'] for d in dados_ordenados]
    
    x = np.arange(len(nomes_seq))
    largura_barra = 0.35

    # =================================================================
    # GRÁFICO 1: Desempenho (mAP vs IDF1) com Barras Agrupadas
    # =================================================================
    plt.figure(figsize=(10, 5))
    plt.bar(x - largura_barra/2, map_scores, width=largura_barra, label='mAP por quadro', color='royalblue')
    plt.bar(x + largura_barra/2, idf1_scores, width=largura_barra, label='IDF1', color='darkorange')
    
    plt.title('Desempenho da Detecção vs Associação (Ordenado por Densidade)', fontsize=14, pad=15)
    plt.ylabel('Pontuação (0 a 1)', fontsize=12)
    plt.xticks(x, nomes_seq, rotation=45, ha='right')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6, axis='y') # Grade apenas horizontal
    
    plt.tight_layout()
    plt.show()

    # =================================================================
    # GRÁFICO 2: Análise de Fracasso (Fragmentações e IDSW)
    # =================================================================
    plt.figure(figsize=(10, 5))
    plt.bar(x - largura_barra/2, razao_ids, width=largura_barra, label='IDs Previstos / IDs Verdadeiros', color='teal')
    plt.bar(x + largura_barra/2, razao_idsw, width=largura_barra, label='IDSW / IDs Verdadeiros', color='crimson')
    
    plt.axhline(1.0, color='black', linestyle='-.', alpha=0.5, label='Referência (Razão Ideal = 1)')
    
    plt.title('Quantificação do Fracasso: Fragmentações e Trocas de ID', fontsize=14, pad=15)
    plt.ylabel('Razão / Proporção', fontsize=12)
    plt.xticks(x, nomes_seq, rotation=45, ha='right')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    plt.tight_layout()
    plt.show()


import cv2
import matplotlib.pyplot as plt

def plotar_tira_de_quadros(video_frames, gt, predicoes, quadros_alvo):
    """
    Plota uma tira de quadros mostrando GT (verde) e Predição (vermelho)
    quadros_alvo: lista de inteiros com os frames que deseja visualizar (ex: buraco de oclusão)
    """
    num_quadros = len(quadros_alvo)
    fig, axes = plt.subplots(1, num_quadros, figsize=(4 * num_quadros, 4))
    if num_quadros == 1:
        axes = [axes]
        
    for idx, f in enumerate(quadros_alvo):
        # Pega a imagem original (f-1 pois o index é 0-based e o frame é 1-based)
        img = video_frames[f-1].copy()
        
        # Filtra GT e Preds para o frame específico
        gt_f = [d for d in gt if int(d[0]) == f]
        pr_f = [d for d in predicoes if int(d[0]) == f]
        
        # Desenha GT em Verde
        for g in gt_f:
            obj_id, x, y, w, h = int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"GT:{obj_id}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
        # Desenha Predição em Vermelho
        for p in pr_f:
            obj_id, x, y, w, h = int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"PR:{obj_id}", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            
        axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"Quadro {f}")
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.show()