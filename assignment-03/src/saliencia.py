"""Parte 3 - Galeria de falhas e atalhos: selecao de casos, Grad-CAM e mapa de oclusao.

Tudo o que e logica fica aqui; o notebook so orquestra e mostra resultados.
Nada aqui treina ou escolhe modelo: o test_public e usado apenas para ANALISAR o modelo final.
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from src.autosup import contar_parametros, desnormalizar

LADO = 224


# ----------------------------------------------------------------------------
# Modelo e previsoes
# ----------------------------------------------------------------------------
def carregar_classificador(state_path: str, n_classes: int, device) -> nn.Module:
    """ResNet-18 com fc de n_classes, carregada de um checkpoint salvo por train_epoch (modo eval)."""
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    m.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    print(f"Parametros do modelo: {contar_parametros(m):,} (limite: 12.000.000)")
    return m.to(device).eval()


@torch.no_grad()
def prever_probs(modelo: nn.Module, loader, device) -> np.ndarray:
    """Probabilidades softmax (N, n_classes), na ordem do loader (use shuffle=False)."""
    modelo.eval()
    return np.concatenate([F.softmax(modelo(x.to(device)), dim=1).cpu().numpy() for x, _ in loader])


# ----------------------------------------------------------------------------
# Selecao dos casos
# ----------------------------------------------------------------------------
def _classes_distintas(indices, y, k):
    escolhidos, vistas = [], set()
    for i in indices:
        if y[i] not in vistas:
            escolhidos.append(int(i))
            vistas.add(y[i])
        if len(escolhidos) == k:
            break
    return escolhidos


def selecionar_erros(probs: np.ndarray, y: np.ndarray, k: int = 3) -> list:
    """Os k erros MAIS CONFIANTES, de classes verdadeiras distintas.
    Sao os mais instrutivos: o modelo estava 'certo' e errado, o que aponta atalhos."""
    conf = probs.max(1)
    erros = np.where(probs.argmax(1) != y)[0]
    return _classes_distintas(erros[np.argsort(-conf[erros])], y, k)


def selecionar_acertos(probs: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> list:
    """k acertos sorteados (seed fixa), de classes distintas. Sorteio, e nao os mais confiantes,
    para nao escolher so os casos bonitos."""
    acertos = np.where(probs.argmax(1) == y)[0]
    np.random.default_rng(seed).shuffle(acertos)
    return _classes_distintas(acertos, y, k)


# ----------------------------------------------------------------------------
# Imagem como o modelo a ve
# ----------------------------------------------------------------------------
def carregar_tensor(raiz: str, image_id: str, transform, device):
    """Devolve (tensor 1x3x224x224 no device, imagem original RGB uint8)."""
    img = cv2.cvtColor(cv2.imread(os.path.join(raiz, image_id + ".jpg")), cv2.COLOR_BGR2RGB)
    x = transform(image=img)["image"].unsqueeze(0).to(device)
    return x, img


def retangulo_conteudo(h: int, w: int, lado: int = LADO):
    """(x0, y0, x1, y1) ocupado pela foto apos LongestMaxSize + PadIfNeeded; o resto e faixa preta."""
    esc = lado / max(h, w)
    nh, nw = int(round(h * esc)), int(round(w * esc))
    y0, x0 = (lado - nh) // 2, (lado - nw) // 2
    return x0, y0, x0 + nw, y0 + nh


def croma_medio(img_rgb: np.ndarray) -> float:
    """Diferenca media entre o maior e o menor canal. Perto de 0 = foto infravermelha (cinza)."""
    c = img_rgb.astype(np.int16)
    return float((c.max(2) - c.min(2)).mean())


# ----------------------------------------------------------------------------
# Mapas de saliencia
# ----------------------------------------------------------------------------
def normalizar_mapa(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def grad_cam(modelo: nn.Module, x: torch.Tensor, classe: int, camada: nn.Module = None) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017) para `classe`, em [0, 1], com o tamanho de x.
    Por padrao usa `modelo.layer4` (grade 7x7 numa entrada de 224: grosseiro para animais pequenos)."""
    modelo.eval()
    camada = modelo.layer4 if camada is None else camada
    guardado = {}

    def ao_passar(_, __, saida):
        guardado["ativ"] = saida
        saida.register_hook(lambda g: guardado.__setitem__("grad", g))

    h = camada.register_forward_hook(ao_passar)
    try:
        modelo.zero_grad(set_to_none=True)
        modelo(x)[0, classe].backward()
    finally:
        h.remove()
    a, g = guardado["ativ"].detach(), guardado["grad"]
    pesos = g.mean(dim=(2, 3), keepdim=True)                 # importancia de cada canal
    cam = F.relu((pesos * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return normalizar_mapa(cam.cpu().numpy())


@torch.no_grad()
def mapa_oclusao(modelo: nn.Module, x: torch.Tensor, classe: int, patch: int = 32,
                 passo: int = 16, batch: int = 64) -> np.ndarray:
    """Oclusao (Zeiler & Fergus, 2014): queda na probabilidade de `classe` ao cobrir cada janela
    patch x patch com a media do dataset (0 apos normalizar). Sem normalizar: o maximo do mapa
    e a maior queda de probabilidade causada por uma unica janela. Resolucao bem melhor que o Grad-CAM."""
    modelo.eval()
    p0 = F.softmax(modelo(x), dim=1)[0, classe].item()
    H, W = x.shape[-2:]
    posicoes = [(i, j) for i in range(0, H - patch + 1, passo) for j in range(0, W - patch + 1, passo)]
    soma, cont = torch.zeros(H, W), torch.zeros(H, W)
    for ini in range(0, len(posicoes), batch):
        bloco = posicoes[ini:ini + batch]
        xs = x.repeat(len(bloco), 1, 1, 1)
        for k, (i, j) in enumerate(bloco):
            xs[k, :, i:i + patch, j:j + patch] = 0.0
        p = F.softmax(modelo(xs), dim=1)[:, classe].cpu()
        for k, (i, j) in enumerate(bloco):
            soma[i:i + patch, j:j + patch] += p0 - p[k]
            cont[i:i + patch, j:j + patch] += 1
    return np.clip((soma / cont.clamp(min=1)).numpy(), 0, None)


def metricas_mapa(mapa: np.ndarray, retang) -> dict:
    """Resume o mapa so dentro da foto (sem a faixa preta do letterbox).
    area_50: fracao da foto necessaria para reunir 50% da massa do mapa (pequena = foco concentrado).
    pico: (x, y) do maximo, nas mesmas coordenadas 0-224 dos eixos da galeria."""
    x0, y0, x1, y1 = retang
    m = mapa[y0:y1, x0:x1]
    acum = np.cumsum(np.sort(m.ravel())[::-1]) / (m.sum() + 1e-12)
    iy, ix = np.unravel_index(m.argmax(), m.shape)
    return {"area_50": float((np.searchsorted(acum, 0.5) + 1) / m.size), "pico": (int(ix + x0), int(iy + y0))}


# ----------------------------------------------------------------------------
# Galeria
# ----------------------------------------------------------------------------
def _sobrepor(ax, img224, mapa, titulo, pico=None):
    ax.imshow(img224)
    ax.imshow(mapa, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    if pico is not None:
        ax.plot(*pico, marker="+", color="white", markersize=14, markeredgewidth=2)
    ax.set_title(titulo, fontsize=9)
    ax.axis("off")


def galeria(ids, y, modelo, raiz, device, classes, transform, titulo="",
            com_classe_verdadeira=False, patch=32, passo=16):
    """Uma linha por imagem: [foto como o modelo a ve | Grad-CAM | oclusao | (Grad-CAM da classe verdadeira)].
    Os mapas explicam a classe PREVISTA; o painel extra mostra onde o modelo olharia para a classe certa.
    Eixos da primeira coluna estao em pixels de 224x224, para estimar o tamanho do animal.
    Devolve (figura, DataFrame com os numeros de cada caso)."""
    ncol = 4 if com_classe_verdadeira else 3
    fig, axes = plt.subplots(len(ids), ncol, figsize=(3.4 * ncol, 3.4 * len(ids)), squeeze=False)
    linhas = []
    for r, (iid, verd) in enumerate(zip(ids, y)):
        x, img = carregar_tensor(raiz, iid, transform, device)
        with torch.no_grad():
            p = F.softmax(modelo(x), dim=1)[0].cpu().numpy()
        prev, conf = int(p.argmax()), float(p.max())
        img224 = desnormalizar(x[0])
        retang = retangulo_conteudo(*img.shape[:2])

        cam = grad_cam(modelo, x, prev)
        ocl_bruto = mapa_oclusao(modelo, x, prev, patch, passo)
        m_cam, m_ocl = metricas_mapa(cam, retang), metricas_mapa(ocl_bruto, retang)

        ok = prev == int(verd)
        ax = axes[r, 0]
        ax.imshow(img224)
        ax.set_xticks(range(0, LADO + 1, 56)); ax.set_yticks(range(0, LADO + 1, 56))
        ax.tick_params(labelsize=7)
        ax.set_title(f"{iid}\nverdadeira: {classes[verd]}\nprevista: {classes[prev]} ({conf:.0%})",
                     fontsize=9, color="tab:green" if ok else "tab:red")
        _sobrepor(axes[r, 1], img224, cam, f"Grad-CAM ({classes[prev]})", m_cam["pico"])
        _sobrepor(axes[r, 2], img224, normalizar_mapa(ocl_bruto), f"Oclusão {patch}px ({classes[prev]})", m_ocl["pico"])
        if com_classe_verdadeira:
            _sobrepor(axes[r, 3], img224, grad_cam(modelo, x, int(verd)), f"Grad-CAM ({classes[verd]}, verdadeira)")

        linhas.append({
            "image_id": iid, "verdadeira": classes[verd], "prevista": classes[prev], "confianca": conf,
            "p_verdadeira": float(p[verd]), "croma_medio": croma_medio(img),
            "tam_original": f"{img.shape[1]}x{img.shape[0]}",
            "escala_p/224": LADO / max(img.shape[:2]),
            "gradcam_area50": m_cam["area_50"], "gradcam_pico": m_cam["pico"],
            "oclusao_area50": m_ocl["area_50"], "oclusao_pico": m_ocl["pico"],
            "oclusao_queda_max": float(ocl_bruto.max()),
        })
    if titulo:
        fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout()
    return fig, pd.DataFrame(linhas)
