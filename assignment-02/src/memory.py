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