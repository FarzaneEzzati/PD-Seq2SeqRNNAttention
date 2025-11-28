import torch
from torch import nn
import torch.nn.functional as F
import random

from Encoder import Encoder
from Decoder import Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Seq2Seq(nn.Module):
    
    def __init__(self, input_seq_len, output_seq_len, input_dim, output_dim, n_layers,
                 enc_hidden_dim, dec_hidden_dim,
                 dropout):
        self.n_layers = n_layers
        self.output_seq_len = output_seq_len
        super().__init__()
        
        self.encoder = Encoder(seq_len=input_seq_len, input_dim=input_dim, n_layers=n_layers,
                               enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim,
                               dropout=dropout)
        
        self.decoder = Decoder(output_dim=output_dim, n_layers=n_layers,
                               enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim,
                               dropout=dropout)
        
        
    def forward(self, inputs, targets, tf_ratio):
        ''' inputs:  [batch_size, input_seq_len, input_dim]
            targets: [batch_size, seq_len] '''
            
        ###########################  1. ENCODER  ##############################
        h = self.encoder.init_hidden(inputs.size(0))  # [batch_size, n_layers, enc_hidden_dim]

        # calls forward method.
        enc_hidden_per_time, enc_hidden_per_layer = self.encoder(inputs, h)
        # [batch_size, seq_len, hidden_dim], [batch_size, n_layer, hidden_dim]
            
        ###########################  2. DECODER  ##############################
        dec_hidden = enc_hidden_per_layer     # [batch_size, n_layers, dec_hidden_dim]

        output_seq_len = self.output_seq_len

        batch_size = inputs.size(0)
        output_dim = self.decoder.output_dim
        preds = torch.zeros(batch_size, output_seq_len, output_dim).to(device)
        for t in range(0, output_seq_len):
            target = targets[:, t] if random.random() < tf_ratio else preds[:, t-1, 1] if t != 0 else preds[:, t, 1]
            pred, dec_hidden = self.decoder(target, dec_hidden, enc_hidden_per_time)  # [batch_size, output_dim]
            preds[:, t, :] = pred  # output: [batch_size, t, output_dim]

        return preds  # [batch_size, trg_seq_len, output_dim]
    