import torch
from torch import nn


class Encoder(nn.Module):
    
    def __init__(self, seq_len, input_dim, n_layers,
                 enc_hidden_dim, dec_hidden_dim, dropout):
        super().__init__()
        self.seq_len = seq_len  # length of each sample (T)
        self.input_dim = input_dim  # count of features for each time step (layer)
        self.dropout = dropout  # dropout rate for generalization purposes
        self.n_layers = n_layers  # count of GRU layers
        self.enc_hidden_dim = enc_hidden_dim  # dim of the hidden states in encoder
        self.dec_hidden_dim = dec_hidden_dim  # dim of the hidden states in decoder

        self.GRU = nn.GRU(input_size=input_dim, hidden_size=enc_hidden_dim,
                          num_layers=n_layers,
                          bidirectional=True, batch_first=True,
                          dropout=dropout)  # GRU network
        self.fc_linear = nn.Linear(enc_hidden_dim, dec_hidden_dim)
        self.fc_sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, src, h):
        """
        src: Input tensor with shape [batch_size, seq_len, input_dim]
        """
        hidden_per_time, hidden_per_layer = self.GRU(src, h)
        # hidden_per_time: GRU outputs at each time step [batch_size, seq_len, 2 * hidden_dim].
        # hidden_per_layer: hidden states from all layers and directions [2 * n_layers, b, hidden_dim]

        hidden_per_time = self.dropout(hidden_per_time)
        hidden_per_layer = hidden_per_layer.permute(1, 0, 2)  # [batch_size, 2 * n_layer, hidden_dim]

        # Summing forward and backward: final output [batch_size, n_layers, hidden_dim]
        for i in range(0, -2*self.n_layers, -2):
            if i == 0:
                hidden_per_layer_sum = hidden_per_layer[:, i-1, :] + hidden_per_layer[:, i-2, :]  # [batch_size, hidden_dim]
                hidden_per_layer_sum = hidden_per_layer_sum.unsqueeze(1)
            else:
                hidden_per_layer_sum = torch.cat((hidden_per_layer_sum,
                                            (hidden_per_layer[:, i-1, :] + hidden_per_layer[:, i-2, :]).unsqueeze(1)), dim=1)

        hidden_per_layer_sum = self.fc_linear(hidden_per_layer_sum)
        hidden_per_layer_sum = self.fc_sigmoid(hidden_per_layer_sum)
        return hidden_per_time, hidden_per_layer_sum   # [batch_size, seq_len, hidden_dim], [batch_size, n_layer, hidden_dim]
        
    
    def init_hidden(self, batch_size):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight = next(self.parameters()).data  # gets the first layer of parameters
        # builds a tensor with the same shape and type of first layer parameters
        h = weight.new(2*self.n_layers, batch_size, self.enc_hidden_dim).zero_().to(device)
        return h
