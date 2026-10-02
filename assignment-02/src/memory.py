import torch
import torch.nn as nn


#  Parte 2
class MovimentoRNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=1):
        """
        Modelo temporal para regressão de Bounding Boxes.
        input_dim=4 corresponde ao vetor geométrico [x, y, w, h].
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # RNN para codificar o histórico temporal do movimento
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Regressor linear para mapear o estado oculto de volta para a geometria [x, y, w, h]
        self.regressor = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, h=None):
        """
        x: Tensor de entrada com shape (batch, seq_len, 4)
        h: Estado oculto anterior com shape (num_layers, batch, hidden_dim)
        
        Retorna:
        - pred_box: Caixa prevista para o instante seguinte (batch, seq_len, 4)
        - h_new: Novo estado oculto atualizado
        """
        out, h_new = self.rnn(x, h)
        pred_box = self.regressor(out)
        
        return pred_box, h_new


import torch
import matplotlib.pyplot as plt

def analisar_gradiente_horizonte(modelo, seq_len=20, device='cpu'):
    """
    Calcula e plota a norma do gradiente dL_T / dh_{T-k}
    para medir analiticamente o horizonte de memória (Parte 4.1).
    """
    modelo.train()
    
    # Cria uma sequência sintética de caixas [batch=1, seq_len, features=4]
    x_seq = torch.randn(1, seq_len, 4, requires_grad=True).to(device)
    h_states = []
    h = None
    
    # Roda a RNN passo a passo para podermos registrar (retain_grad) cada h_t
    for t in range(seq_len):
        x_t = x_seq[:, t:t+1, :]
        pred, h = modelo(x_t, h)
        h.retain_grad()
        h_states.append(h)
        
    # Finge uma loss qualquer no último instante T
    loss = pred.sum()
    loss.backward()
    
    # Coleta a norma dos gradientes extraídos
    normas = []
    distancias_k = []
    
    T = seq_len - 1
    for t in range(seq_len):
        # A distância para o instante final é k = T - t
        k = T - t
        
        # VALIDAÇÃO APLICADA: Evita erro de 'NoneType' no último estado oculto (h_T), 
        # que recebe gradiente nulo por ser o fim da propagação.
        grad = h_states[t].grad
        grad_norm = grad.norm().item() if grad is not None else 0.0
        
        normas.append(grad_norm)
        distancias_k.append(k)
        
    # Plota a curva invertendo para o eixo X ser a distância K (de 0 até seq_len)
    plt.figure(figsize=(8, 4))
    plt.plot(distancias_k[::-1], normas[::-1], marker='o', color='purple')
    plt.xlabel("Distância de tempo no passado (k quadros)")
    plt.ylabel("Norma do Gradiente (dL_T / dh_{T-k})")
    plt.title("Horizonte Analítico de Memória (Vanishing Gradient)")
    plt.grid(True)
    plt.show()