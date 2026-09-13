import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader

# Garante que o Python reconheça a pasta raiz (uma acima de src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import UNetTernary
from src.dataset import DSB2018TernaryDataset, collate_fn_ternary
from src.evaluating import evaluate_instances_ternary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/stage1_train')
    parser.add_argument('--weights', type=str, default='docs/models/best_model_ternary.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu')
    
    print(f"Carregando modelo e pesos de: {args.weights}")
    model = UNetTernary().to(device) #[cite: 5]
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    
    dataset = DSB2018TernaryDataset(args.data_dir) #[cite: 2]
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn_ternary) #[cite: 2]

    print("Avaliando instâncias via Watershed...")
    aps, count_errors, densities, per_image_aps, samples = evaluate_instances_ternary(
        model=model, 
        loader=loader,
        interior_thresh=0.3, #[cite: 3]
        fg_thresh=0.5, #[cite: 3]
        min_marker_size=2 #[cite: 3]
    ) #[cite: 3]
    
    mAP = sum(per_image_aps) / len(per_image_aps) if len(per_image_aps) > 0 else 0
    mean_count_err = sum(count_errors) / len(count_errors) if len(count_errors) > 0 else 0
    
    print(f"\n======================================")
    print(f"mAP [0.5:0.95]: {mAP:.4f}")
    print(f"Erro Absoluto de Contagem: {mean_count_err:.2f}")
    print(f"======================================")