
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