import sys
import os
import argparse

# Garante que o Python reconheça a pasta raiz (uma acima de src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DSB2018TernaryDataset
from src.functions import ablation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/stage1_train')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='docs/models')
    args = parser.parse_args()

    print(f"Carregando dataset de: {args.data_dir}")
    dataset = DSB2018TernaryDataset(args.data_dir) #[cite: 2]
    
    print("Iniciando treinamento (Ablação)...")
    df_summary, df_runs = ablation(
        train_dataset=dataset, 
        val_dataset=dataset, 
        epochs=args.epochs,
        save_dir=args.save_dir
    ) #[cite: 4]
    
    print("\nTreinamento concluído. Resumo:")
    print(df_summary)