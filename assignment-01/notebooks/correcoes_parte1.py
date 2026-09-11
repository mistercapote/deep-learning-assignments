
# ---------------------------------------------------------------------------
# 1. evaluate_instances corrigida
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_instances(model, loader, prob_threshold=0.5, min_size=5, k_samples=6):
    model.eval()

    n_threshold = len(IOU_THRESHOLDS)
    tp_total = np.zeros(n_threshold)
    denom_total = np.zeros(n_threshold)

    count_errors = []
    densities = []
    per_image_aps = []   # NOVO: mAP (média nos 10 limiares) de cada imagem
    samples = []

    for images, _, instance_masks_batch, _ in tqdm(loader, desc='Avaliando instâncias'):
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        for b in range(images.size(0)):
            pred_masks = extract_instances(probs[b, 0], prob_threshold, min_size)
            gt_masks = instance_masks_batch[b]
            n_pred, n_gt = len(pred_masks), len(gt_masks)
            count_error = abs(n_pred - n_gt)
            count_errors.append(count_error)
            densities.append(n_gt)

            img_plot = images[b].cpu().numpy().transpose(1, 2, 0)
            samples.append((count_error, img_plot, gt_masks, pred_masks))

            if n_pred == 0 and n_gt == 0:
                per_image_aps.append(1.0)
                continue

            if n_pred == 0 or n_gt == 0:
                denom_total += n_pred + n_gt
                per_image_aps.append(0.0)
                continue

            pairs = []
            for i, m1 in enumerate(pred_masks):
                for j, m2 in enumerate(gt_masks):
                    inter = np.logical_and(m1, m2).sum()
                    union = np.logical_or(m1, m2).sum()
                    pairs.append((inter / union if union != 0 else 0.0, i, j))
            pairs.sort(key=lambda x: -x[0])

            img_aps = np.zeros(n_threshold)
            for k, t in enumerate(IOU_THRESHOLDS):
                matched_pred, matched_gt = set(), set()
                tp = 0
                for iou, i, j in pairs:
                    if iou < t:
                        break
                    if i not in matched_pred and j not in matched_gt:
                        matched_pred.add(i)
                        matched_gt.add(j)
                        tp += 1
                fp = n_pred - tp
                fn = n_gt - tp
                denom = tp + fp + fn
                img_aps[k] = tp / denom if denom > 0 else 1.0

                tp_total[k] += tp
                denom_total[k] += denom

            per_image_aps.append(float(img_aps.mean()))

    aps = np.where(denom_total > 0, tp_total / denom_total, 1.0)
    mAP = float(np.mean(aps))

    # CORREÇÃO: ordena pelas PIORES (maior erro de contagem primeiro) antes
    # de truncar -- antes disso, samples[:k_samples] pegava as primeiras
    # imagens do loader (ordem arbitrária do dataset), não as piores.
    samples.sort(key=lambda s: -s[0])

    return mAP, aps, count_errors, densities, per_image_aps, samples[:k_samples]


# ---------------------------------------------------------------------------
# 2. Curvas de treino e AP por limiar (extraídas das células soltas)
# ---------------------------------------------------------------------------

def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'])
    axes[0].set_title('Loss de treino (BCE + Dice)')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['val_iou'], label='IoU')
    axes[1].plot(history['val_dice'], label='Dice')
    axes[1].set_title('Métricas de validação (nível de pixel)')
    axes[1].set_xlabel('Época')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ap_by_threshold(aps, thresholds=None):
    thresholds = IOU_THRESHOLDS if thresholds is None else thresholds
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, aps, marker='o')
    plt.xlabel('Limiar de IoU')
    plt.ylabel('AP')
    plt.title('AP por limiar de IoU')
    plt.grid(alpha=0.3)
    plt.show()


# ---------------------------------------------------------------------------
# 3. Item 5 — fracasso vs. densidade, agora com binning (tendência visível)
# ---------------------------------------------------------------------------

def plot_density_failure(densities, count_errors, per_image_aps=None, n_bins=6):
    """
    Gráfico de dispersão bruto (cinza, transparente) + curva de médias por
    faixa de densidade (destacada), pra tendência ficar visível mesmo com
    ruído entre imagens individuais. Se `per_image_aps` for passado, plota
    um segundo painel com mAP-por-imagem vs. densidade.
    """
    densities = np.asarray(densities)
    count_errors = np.asarray(count_errors)

    # bins por quantil: aprox. mesmo número de imagens por faixa
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(densities, quantiles))
    bin_idx = np.digitize(densities, edges[1:-1], right=True)
    n_bins_eff = len(edges) - 1

    bin_density = [densities[bin_idx == k].mean() for k in range(n_bins_eff)]
    bin_error = [count_errors[bin_idx == k].mean() for k in range(n_bins_eff)]

    n_panels = 2 if per_image_aps is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    axes = [axes] if n_panels == 1 else axes

    axes[0].scatter(densities, count_errors, alpha=0.25, color='gray',
                     label='imagens individuais')
    axes[0].plot(bin_density, bin_error, marker='o', color='tomato',
                 linewidth=2, label='média por faixa')
    axes[0].set_xlabel('Nº de núcleos (densidade, ground truth)')
    axes[0].set_ylabel('Erro absoluto de contagem')
    axes[0].set_title('Erro de contagem vs. densidade de objetos')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if per_image_aps is not None:
        per_image_aps = np.asarray(per_image_aps)
        bin_ap = [per_image_aps[bin_idx == k].mean() for k in range(n_bins_eff)]
        axes[1].scatter(densities, per_image_aps, alpha=0.25, color='gray',
                         label='imagens individuais')
        axes[1].plot(bin_density, bin_ap, marker='o', color='steelblue',
                     linewidth=2, label='média por faixa')
        axes[1].set_xlabel('Nº de núcleos (densidade, ground truth)')
        axes[1].set_ylabel('mAP da imagem (0.50:0.95)')
        axes[1].set_title('mAP vs. densidade de objetos')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    corr_err = np.corrcoef(densities, count_errors)[0, 1]
    print(f'Correlação (densidade x erro de contagem): {corr_err:.3f}')
    if per_image_aps is not None:
        corr_ap = np.corrcoef(densities, per_image_aps)[0, 1]
        print(f'Correlação (densidade x mAP por imagem): {corr_ap:.3f}')


# ---------------------------------------------------------------------------
# 4. Amostras qualitativas, agora coloridas por instância
# ---------------------------------------------------------------------------

def _distinct_colors(n):
    """
    Gera n cores RGB distintas espaçando matizes (hue) no espectro HSV.
    Diferente de colormaps discretos (tab20, tem só 20 cores e repete),
    isso escala pra qualquer número de instâncias sem repetir cor.
    """
    if n == 0:
        return []
    return [colorsys.hsv_to_rgb(i / n, 0.85, 0.95) for i in range(n)]


def colorize_instances(masks, shape, alpha=0.6):
    """
    masks: lista de máscaras binárias (H,W), uma por instância.
    shape: (H,W) da imagem.
    Retorna um overlay RGBA (H,W,4): cada instância pintada com uma cor
    distinta; fundo transparente (alpha=0) onde não há nenhuma instância.
    """
    H, W = shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    colors = _distinct_colors(len(masks))
    for m, color in zip(masks, colors):
        overlay[m.astype(bool)] = (*color, alpha)
    return overlay


def plot_samples(samples, n_show=6):
    """
    samples: lista de tuplas (count_error, img, gt_masks, pred_masks),
    como retornado por evaluate_instances. Mostra 3 colunas por amostra:
    imagem original | GT colorido por instância | Predição colorida por instância.
    """
    n_show = min(len(samples), n_show)
    fig, axes = plt.subplots(n_show, 3, figsize=(10, 3 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for row, (count_error, img, gt_masks, pred_masks) in enumerate(samples[:n_show]):
        H, W = img.shape[:2]
        gt_overlay = colorize_instances(gt_masks, (H, W))
        pred_overlay = colorize_instances(pred_masks, (H, W))

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f'Imagem (erro={count_error})')

        axes[row, 1].imshow(img)
        axes[row, 1].imshow(gt_overlay)
        axes[row, 1].set_title(f'GT ({len(gt_masks)} inst.)')

        axes[row, 2].imshow(img)
        axes[row, 2].imshow(pred_overlay)
        axes[row, 2].set_title(f'Pred ({len(pred_masks)} inst.)')

        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
