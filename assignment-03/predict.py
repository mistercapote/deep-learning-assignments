import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from src.redimensionamento import DatasetTrap, transform_val

def localizar_arquivo_classes(caminho_informado: str | None) -> Path:
    base = Path(__file__).resolve().parent
    candidatos = [Path(caminho_informado)] if caminho_informado else []
    candidatos.extend([
        base / "classes.csv",
        base / "pa3_dados" / "classes.csv",
        base / "pa3_dados" / "labels" / "classes.csv",
    ])
    for caminho in candidatos:
        if caminho.is_file():
            return caminho
    raise FileNotFoundError("Arquivo classes.csv não encontrado.")


def carregar_nomes_classes(caminho_classes: Path) -> list[str]:
    """Lê os nomes das classes em ordem alfabética (mesma ordem do LabelEncoder do treino)."""
    df = pd.read_csv(caminho_classes)
    if "label" in df.columns:
        nomes = df["label"].dropna().astype(str).unique().tolist()
    elif "class" in df.columns:
        nomes = df["class"].dropna().astype(str).unique().tolist()
    else:
        nomes = df.iloc[:, 0].dropna().astype(str).unique().tolist()
    return sorted(nomes)


def carregar_modelo(caminho_checkpoint: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Instancia a ResNet-18 sem baixar pesos (offline) e carrega o checkpoint treinado."""
    modelo = resnet18(weights=None)
    modelo.fc = nn.Linear(modelo.fc.in_features, num_classes)

    state_dict = torch.load(caminho_checkpoint, map_location=device, weights_only=True)
    modelo.load_state_dict(state_dict)
    modelo.to(device)
    modelo.eval()
    return modelo


def main():
    parser = argparse.ArgumentParser(
        description="Inferência PA3 — Classificação de fauna em câmeras trap (CPU / offline)."
    )
    parser.add_argument(
        "--imagens",
        type=str,
        required=True,
        help="Pasta contendo as imagens (.jpg) para predição.",
    )
    parser.add_argument(
        "--saida",
        type=str,
        required=True,
        help="Caminho do arquivo CSV de saída (colunas: image_id,label).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/modelo_final.pt",
        help="Caminho para o checkpoint (.pt) do modelo final.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Caminho para o arquivo classes.csv.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do batch na inferência em CPU.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    pasta_imagens = Path(args.imagens)
    caminho_saida = Path(args.saida)
    caminho_checkpoint = Path(args.checkpoint)
    caminho_classes = localizar_arquivo_classes(args.classes)

    if not pasta_imagens.is_dir():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {pasta_imagens}")
    if not caminho_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {caminho_checkpoint}")
    if not caminho_classes.is_file():
        raise FileNotFoundError(f"Arquivo de classes não encontrado: {caminho_classes}")

    # 1. Nomes das classes (mesma ordem alfabética usada pelo LabelEncoder no notebook)
    nomes_classes = carregar_nomes_classes(caminho_classes)
    # 2. Carrega modelo em CPU sem acesso à internet e imprime o número de parâmetros
    modelo = carregar_modelo(caminho_checkpoint, len(nomes_classes), device)
    num_parametros = sum(p.numel() for p in modelo.parameters())
    print(f"Número de parâmetros do modelo: {num_parametros:,}")

    # 3. Lista as imagens da pasta e reaproveita o DatasetTrap + transform_val do projeto
    arquivos_img = sorted(pasta_imagens.glob("*.jpg"))
    if not arquivos_img:
        raise RuntimeError(f"Nenhuma imagem .jpg encontrada em: {pasta_imagens}")

    image_ids = [p.stem for p in arquivos_img]
    df_infer = pd.DataFrame({"image_id": image_ids, "label": 0})

    dataset = DatasetTrap(df=df_infer, raiz=str(pasta_imagens), augment=transform_val)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 4. Inferência em CPU
    predicoes_idx = []
    with torch.no_grad():
        for imagens, _ in loader:
            imagens = imagens.to(device)
            logits = modelo(imagens)
            preds = logits.argmax(dim=1).cpu().tolist()
            predicoes_idx.extend(preds)

    # 5. Mapeia índices de volta para os nomes de classes.csv e salva o CSV
    labels_preditos = [nomes_classes[i] for i in predicoes_idx]
    df_saida = pd.DataFrame({
        "image_id": image_ids,
        "label": labels_preditos,
    })

    if caminho_saida.parent != Path(""):
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    df_saida.to_csv(caminho_saida, index=False)
    print(f"Predições salvas em: {caminho_saida} ({len(df_saida)} imagens)")


if __name__ == "__main__":
    main()