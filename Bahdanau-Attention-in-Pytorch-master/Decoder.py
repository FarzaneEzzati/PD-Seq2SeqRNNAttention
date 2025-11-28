import torch
from torch import nn
import torch.nn.functional as tnnf
from Attention import Attention


class Decoder(nn.Module):
    
    def __init__(self, output_dim, n_layers,
                 enc_hidden_dim, dec_hidden_dim, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.dec_hidden_dim = dec_hidden_dim  # dim of hidden states
        self.n_layers = n_layers

        self.GRU = nn.GRU(input_size=1 + 2 * enc_hidden_dim, hidden_size=dec_hidden_dim, num_layers=n_layers,
                          bidirectional=False, batch_first=True,
                          dropout=dropout)

        # cal. attention scores between the encoder's hidden states and the decoder's hidden states
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        
        self.fc_linear = nn.Linear(1 + 2*enc_hidden_dim + dec_hidden_dim, output_dim)
        self.fc_tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, target, dec_hidden, enc_hidden_per_time):
        '''
            target:         [batch_size]
            dec_hidden:     [batch_size, n_layers, dec_hidden_dim]      (1st hidden_dec = encoder's last_h's last layer)
            hidden_per_time: [batch_size, seq_len, n_hidden_enc * 2]   (* 2 if bi-directional)
            --> not to be confused with encoder's last hidden state.
            last_layer_enc is ALL hidden states (of time-steps 0,1,...,t-1) of the last LAYER.
        '''
        target = target.unsqueeze(1)
        ########################### ATTENTION #########################
        att_weights = self.attention(dec_hidden, enc_hidden_per_time)  # [batch_size, input_seq_len]
        att_weights = att_weights.unsqueeze(1)   # [batch_size, 1, input_seq_len]

        # calculate weight sum
        weighted_sum = torch.bmm(att_weights, enc_hidden_per_time)  # [batch_size, 1, enc_hidden_dim]

        ########################### GRU #########################
        gru_input = torch.cat((target.unsqueeze(2), weighted_sum), dim=2)  # [batch_size, 1, target_dim + enc_hidden_dim]
        hidden_per_time, hidden_per_layer = self.GRU(gru_input, dec_hidden.permute(1, 0, 2))
        # hidden_per_time: GRU outputs at each time step [batch_size, seq_len, hidden_dim].
        # hidden_per_layer: hidden states from all layers and directions [n_layers, b, hidden_dim]
        hidden_per_layer = hidden_per_layer.permute(1, 0, 2)  # [batch_size, n_layers, n_hidden_dec]
        
        ########################### 5. FINAL FC LAYER #########################
        fc_in = torch.cat((target,  # [batch_size, 1]
                           weighted_sum.squeeze(1),  # [batch_size, enc_hidden_dim]
                           hidden_per_time.squeeze(1)), dim=1)  # [batch_size, dec_hidden_dim]

        output = self.fc_linear(fc_in)  # [batch_size, output_dim]

        return output, hidden_per_layer  # [batch_size, output_dim], [batch_size, n_layers, n_hidden_dec]
        
