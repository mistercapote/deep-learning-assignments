"""Parte 1 - Trilha A (auto-supervisionado): SimCLR sobre o train inteiro (sem rotulos),
avaliacao linear e fine-tuning com os rotulos do orcamento.

Tudo o que e logica fica aqui; o notebook so orquestra e mostra resultados.
"""
import os
import random
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from src.redimensionamento import eval_model, train_epoch

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def fixar_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def contar_parametros(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def listar_ids(raiz: str) -> list:
    """Ids (sem extensao) de todas as imagens do train. Nenhum rotulo e lido."""
    return sorted(f[:-4] for f in os.listdir(raiz) if f.lower().endswith(".jpg"))


# ----------------------------------------------------------------------------
# Aumentacao e dataset do SimCLR
# ----------------------------------------------------------------------------
def criar_augment_simclr(scale=(0.2, 1.0), ratio=(0.75, 1.3333), lado=224) -> A.Compose:
    """Aumentacao do SimCLR. O parametro `scale` do RandomResizedCrop e a variavel do
    experimento obrigatorio de aumentacao: com animal em ~2% do quadro, um crop de
    20% da imagem frequentemente nao contem o animal."""
    return A.Compose([
        A.RandomResizedCrop(size=(lado, lado), scale=scale, ratio=ratio),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


class DatasetSimCLR(Dataset):
    """Devolve duas views aumentadas da mesma imagem (sem rotulo)."""

    def __init__(self, raiz: str, ids: list, augment: A.Compose) -> None:
        self.raiz, self.ids, self.augment = raiz, ids, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.raiz, self.ids[idx] + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.augment(image=img)["image"], self.augment(image=img)["image"]


def desnormalizar(t: torch.Tensor) -> np.ndarray:
    """Tensor CxHxW normalizado -> array HxWx3 em [0,1], para plotar."""
    m = torch.tensor(MEAN).view(3, 1, 1)
    s = torch.tensor(STD).view(3, 1, 1)
    return (t.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
def criar_encoder(inicializacao: str = "imagenet", state_path: str = None) -> nn.Module:
    """ResNet-18 sem a camada fc (saida: vetor de 512). `inicializacao`: 'imagenet'|'aleatorio'."""
    pesos = ResNet18_Weights.IMAGENET1K_V1 if inicializacao == "imagenet" else None
    m = resnet18(weights=pesos)
    m.fc = nn.Identity()
    if state_path is not None:
        m.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    return m


class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, dim_feat: int = 512, dim_proj: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(dim_feat, dim_feat), nn.ReLU(), nn.Linear(dim_feat, dim_proj))

    def forward(self, x):
        return self.head(self.encoder(x))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperatura: float = 0.5) -> torch.Tensor:
    """NT-Xent: para cada view, o positivo e a outra view da mesma imagem; os 2N-2 demais sao negativos."""
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0).float(), dim=1)
    sim = z @ z.T / temperatura
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    alvo = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, alvo)


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def _passo_simclr(modelo, opt, v1, v2, temperatura, amp, device):
    """Um passo de treino. As duas views vao juntas num unico forward (SimCLR original)."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        z = modelo(torch.cat([v1, v2], dim=0))
    z1, z2 = z.chunk(2)
    loss = nt_xent(z1, z2, temperatura)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss


def _loader_simclr(raiz_train, augment, batch_size, num_workers, max_imgs, seed):
    ids = listar_ids(raiz_train)
    if max_imgs is not None and max_imgs < len(ids):
        ids = sorted(random.Random(seed).sample(ids, max_imgs))
    return DataLoader(DatasetSimCLR(raiz_train, ids, augment), batch_size=batch_size, shuffle=True,
                      drop_last=True, num_workers=num_workers, persistent_workers=num_workers > 0)


def perfilar_pretreino(raiz_train, augment, device, batch_size=128, num_workers=4, n_batches=6, amp=False):
    """Mede onde o tempo vai, em segundos por batch: (1) so o DataLoader (leitura + aumentacao na CPU);
    (2) so o passo de treino, com dados sinteticos (GPU). Imprime tambem a estimativa por epoca."""
    loader = _loader_simclr(raiz_train, augment, batch_size, num_workers, None, 0)
    t0 = time.time()
    it = iter(loader)
    next(it)
    t_subida = time.time() - t0
    t0 = time.time()
    for _ in range(n_batches):
        next(it)
    t_dados = (time.time() - t0) / n_batches

    modelo = SimCLR(criar_encoder("aleatorio")).to(device).train()   # pesos nao mudam o tempo
    opt = torch.optim.AdamW(modelo.parameters(), lr=3e-4)
    lado = augment(image=np.zeros((256, 256, 3), np.uint8))["image"].shape[-1]
    v1 = torch.randn(batch_size, 3, lado, lado, device=device)
    v2 = torch.randn(batch_size, 3, lado, lado, device=device)
    _passo_simclr(modelo, opt, v1, v2, 0.5, amp, device)           # aquecimento
    _sync(device)
    t0 = time.time()
    for _ in range(n_batches):
        _passo_simclr(modelo, opt, v1, v2, 0.5, amp, device)
    _sync(device)
    t_gpu = (time.time() - t0) / n_batches

    n_steps = len(loader)
    print(f"subida dos workers : {t_subida:6.1f} s (uma vez)")
    print(f"dados (CPU)        : {t_dados:6.2f} s/batch")
    print(f"passo (GPU, amp={amp}): {t_gpu:6.2f} s/batch")
    print(f"por epoca ({n_steps} batches): ~{n_steps * max(t_dados, t_gpu) / 60:.1f} min "
          f"-> gargalo: {'DADOS (CPU)' if t_dados > t_gpu else 'GPU'}")
    return {"dados": t_dados, "gpu": t_gpu}


def pretreino_simclr(encoder, raiz_train, augment, device, label,
                     epocas=5, batch_size=128, lr=3e-4, temperatura=0.5,
                     num_workers=4, seed=0, amp=False, max_imgs=None):
    """Continua o pre-treino do encoder com SimCLR sobre o train (sem rotulos).
    amp: bfloat16 no forward (mais rapido em GPU recente). max_imgs: usa so um subconjunto sorteado do train.
    Salva o encoder em models/simclr_{label}.pt e devolve o historico da loss."""
    fixar_seed(seed)
    loader = _loader_simclr(raiz_train, augment, batch_size, num_workers, max_imgs, seed)
    modelo = SimCLR(encoder).to(device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    historico = {"loss": []}

    for ep in range(epocas):
        modelo.train()
        t0, acc = time.time(), 0.0
        for v1, v2 in loader:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            acc += _passo_simclr(modelo, opt, v1, v2, temperatura, amp, device).item() * v1.size(0)
        sched.step()
        historico["loss"].append(acc / (len(loader) * batch_size))
        print(f"[{label}] Epoch {ep + 1}/{epocas} | NT-Xent: {historico['loss'][-1]:.4f} | {time.time() - t0:.0f}s")

    os.makedirs("models", exist_ok=True)
    torch.save(encoder.state_dict(), f"models/simclr_{label}.pt")
    return historico


# ----------------------------------------------------------------------------
# Avaliacao linear (encoder congelado + regressao logistica)
# ----------------------------------------------------------------------------
@torch.no_grad()
def extrair_features(encoder, loader, device):
    encoder.eval().to(device)
    X, y = [], []
    for imgs, rot in loader:
        X.append(encoder(imgs.to(device)).cpu().numpy())
        y.append(rot.numpy())
    return np.concatenate(X), np.concatenate(y)


def avaliacao_linear(encoder, loader_train, loader_val, loader_test, device, Cs=(0.01, 0.1, 1, 10, 100, 1000)):
    """Congela o encoder, treina regressao logistica nos 105 rotulos e escolhe C no VAL (F1 macro).
    Os loaders devem usar transform_val (sem aumentacao) e shuffle=False."""
    Xtr, ytr = extrair_features(encoder, loader_train, device)
    Xva, yva = extrair_features(encoder, loader_val, device)
    Xte, yte = extrair_features(encoder, loader_test, device)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    melhor = {"f1_val": -1}
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
        f1v = f1_score(yva, clf.predict(Xva), average="macro")
        if f1v > melhor["f1_val"]:
            melhor = {"f1_val": f1v, "C": C, "clf": clf}
    prev = melhor["clf"].predict(Xte)
    return {"C": melhor["C"], "f1_val": melhor["f1_val"],
            "f1": f1_score(yte, prev, average="macro"), "acc": accuracy_score(yte, prev),
            "prev": prev, "rot": yte}


# ----------------------------------------------------------------------------
# Fine-tuning com os rotulos do orcamento (reaproveita train_epoch / eval_model da Parte 0)
# ----------------------------------------------------------------------------
def criar_classificador(inicializacao: str, state_path: str, n_classes: int) -> nn.Module:
    m = criar_encoder(inicializacao, state_path)
    m.fc = nn.Linear(512, n_classes)
    return m


def finetune_e_testa(inicializacao, state_path, loader_train, loader_val, loader_test,
                     pesos_tensor, device, label, epocas=5, lr=1e-4, seed=0):
    """Fine-tuning completo; checkpoint pela melhor F1 macro no val; metricas no test_public.
    state_path=None usa so a inicializacao (ImageNet ou aleatoria), sem pre-treino SSL."""
    fixar_seed(seed)
    modelo = criar_classificador(inicializacao, state_path, pesos_tensor.numel()).to(device)
    criterio = nn.CrossEntropyLoss(weight=pesos_tensor)
    hist = train_epoch(model=modelo, device=device, criterio=criterio, train_loader=loader_train,
                       val_loader=loader_val, label=label, epochs=epocas, lr=lr)
    modelo.load_state_dict(torch.load(f"models/best_model_{label}.pt", map_location=device, weights_only=True))
    prev, rot = eval_model(modelo, loader_test, criterio, device)
    return {"hist": hist, "f1": f1_score(rot, prev, average="macro"),
            "acc": accuracy_score(rot, prev), "prev": prev, "rot": rot}
