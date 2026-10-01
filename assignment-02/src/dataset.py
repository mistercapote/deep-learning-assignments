import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrajetoriasMOTDataset(Dataset):
    def __init__(self, gt_data, seq_len=10):
        """
        gt_data: lista retornada pela nossa função carregar_gt() 
                 no formato [frame, id, x, y, w, h, conf].
        seq_len: tamanho da janela temporal de treinamento.
        """
        self.seq_len = seq_len
        self.amostras = []
        
        # 1. Agrupar caixas por ID da trajetória
        trajetorias = {}
        for row in gt_data:
            frame, obj_id, x, y, w, h, conf = row
            if obj_id not in trajetorias:
                trajetorias[obj_id] = []
            # Guarda a geometria (aqui podemos futuramente normalizar os dados)
            trajetorias[obj_id].append([x, y, w, h])
            
        # 2. Criar janelas deslizantes para o treinamento
        # Precisamos de seq_len + 1 para formar os pares de entrada(X) e alvo(Y)
        tamanho_janela = seq_len + 1 
        
        for obj_id, boxes in trajetorias.items():
            if len(boxes) >= tamanho_janela:
                for i in range(len(boxes) - tamanho_janela + 1):
                    janela = boxes[i : i + tamanho_janela]
                    self.amostras.append(janela)
                    
    def __len__(self):
        return len(self.amostras)
        
    def __getitem__(self, idx):
        # Converte para tensor
        janela = torch.tensor(self.amostras[idx], dtype=torch.float32)
        
        # X: Instantes t (de 0 até seq_len-1)
        # Y: Instantes t+1 (de 1 até seq_len)
        x = janela[:-1]
        y = janela[1:]
        
        return x, y