import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        
        super().__init__()
        
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        
        self.W = nn.Linear(2*enc_hidden_dim + dec_hidden_dim, dec_hidden_dim, bias=False)
        self.V = nn.Parameter(torch.rand(dec_hidden_dim))
        
    
    def forward(self, dec_hidden, hidden_per_time):
        ''' 
            PARAMS:           
                dec_hidden:     [batch_size, dec_n_layers, dec_hidden_dim]    (1st hidden_dec = encoder's last_h's last layer)
                enc_last_layer: [batch_size, seq_len, enc_hidden_dime]
            
            RETURN:
                att_weights:    [batch_size, src_seq_len]
            Important: in each time step of decoder the attention weights are calculated.
        '''
        # Get batch size and source length
        # This step aligns the enc_lasy_layer hidden states shape with the same shape for dec_last_layer hidden states
        batch_size = hidden_per_time.size(0)
        src_seq_len = hidden_per_time.size(1)  # the length of input
        
        # Use only the last decoder hidden state
        # Repeat decoder hidden across all source positions
        dec_hidden = dec_hidden[:, -1, :].unsqueeze(1).repeat(1, src_seq_len, 1)  # [batch_size, src_seq_len, dec_hidden_dim]
        
        # Compute energy transformation
        tanh_W_s_h = torch.tanh(self.W(torch.cat((dec_hidden, hidden_per_time), dim=2)))  # [batch_size, src_seq_len, dec_hidden_dim]
        
        # Prepare for dot-product with vector v
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)       # [batch_size, dec_hidden_dim, seq_len]
        
        # Repeat learnable vector v for each batch
        V = self.V.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, dec_hidden_dim]
        
        # Compute unnormalized attention scores and get softmax
        e = torch.bmm(V, tanh_W_s_h).squeeze(1)        # [batch_size, seq_len]
        att_weights = F.softmax(e, dim=1)              # [batch_size, src_seq_len]
        
        return att_weights  # [batch_size, src_seq_len]
    