
import matplotlib.pyplot as plt


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