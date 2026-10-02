import torch.nn as nn
import torch.optim as optim
import torch
from .dataset import TrajetoriasMOTDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np

def treinar_modelo_temporal(modelo, gt_completo, seq_len=10, batch_size=32, epochs=15, lr=1e-3):
    """
    modelo: Instância da nossa MovimentoRNN (ou MovimentoAblacao).
    gt_completo: Lista com o Ground Truth de TODAS as sequências de treino agregadas.
    """
    device = 'cpu'
    modelo.to(device)
    modelo.train()
    
    # Prepara os dados
    dataset = TrajetoriasMOTDataset(gt_completo, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Otimizador e Função de Perda (Smooth L1 conforme a regra)
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    
    print(f"Iniciando treino com {len(dataset)} amostras. Dispositivo: {device}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward: passa a sequência X e prevê Y
            # O estado oculto h inicializa zerado automaticamente
            pred_y, _ = modelo(batch_x)
            
            # Calcula o erro usando Smooth L1
            loss = criterion(pred_y, batch_y)
            
            # Backward
            loss.backward()
            
            # ---> CORREÇÃO AQUI: Gradient Clipping <---
            # Evita a explosão de gradientes (comum em RNNs padrão) 
            # limitando a norma máxima dos gradientes a 1.0 antes do passo do otimizador.
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
        media_loss = epoch_loss / len(dataloader)
        print(f"Época [{epoch+1}/{epochs}] | Perda Smooth-L1: {media_loss:.4f}")
        
    return modelo