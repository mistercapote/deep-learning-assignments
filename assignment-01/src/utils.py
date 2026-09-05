import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(all_mAPs, all_count_errors, all_densities):
    """Imprime as médias globais e plota os gráficos de falha baseados na densidade."""
    print(f"Média global mAP: {np.mean(all_mAPs):.4f}")
    print(f"Média global Erro Absoluto de Contagem: {np.mean(all_count_errors):.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    color = 'tab:blue'
    ax[0].set_xlabel('Densidade de Objetos (Qtd. de Instâncias Reais)')
    ax[0].set_ylabel('mAP (Threshold 0.5 a 0.95)', color=color)
    ax[0].scatter(all_densities, all_mAPs, color=color, alpha=0.6, label='mAP')
    ax[0].tick_params(axis='y', labelcolor=color)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_title("Quantificação de Falhas: Desempenho vs. Densidade", fontsize = 10, fontweight='bold')

    color = 'tab:red'
    ax[1].set_ylabel('Erro Absoluto de Contagem', color=color)
    ax[1].scatter(all_densities, all_count_errors, color=color, alpha=0.6, label='Erro de Contagem')
    ax[1].set_xlabel('Densidade de Objetos (Qtd. de Instâncias Reais)')
    ax[1].tick_params(axis='y', labelcolor=color)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_title("Quantificação de Falhas: Desempenho vs. Densidade", fontsize = 10,  fontweight='bold')
    plt.grid(False)
    fig.tight_layout()
    plt.show()



def plot_samples(samples, part: int):
    """Renderiza uma matriz comparativa de amostras aleatórias."""
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    fig.suptitle(f"Amostra de {len(samples)} Resultados - Parte {part}", fontsize=14, fontweight='bold')

    for idx, (err, img, gt, pred) in enumerate(samples):
        # Imagem Original (Escala de cinza com cmap='gray')
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title("Original")
        axes[idx, 0].axis('off')

        # Gabarito
        # nipy_spectral ajuda a diferenciar instâncias distintas
        axes[idx, 1].imshow(gt, cmap='nipy_spectral', interpolation='nearest')
        axes[idx, 1].set_title(f"Gabarito (Instâncias: {len(np.unique(gt))-1})")
        axes[idx, 1].axis('off')

        # Predição
        axes[idx, 2].imshow(pred, cmap='nipy_spectral', interpolation='nearest')
        axes[idx, 2].set_title(f"Predição (Instâncias: {len(np.unique(pred))-1}) | Erro: {err}")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.show()