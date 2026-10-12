"""Seq2Seq network: GRU encoder + GRU decoder with Bahdanau attention, and the quantile loss.

Seq2Seq (Parent Container)
├── Encoder (Processes source sequence → Hidden states)
└── Decoder (Generates target sequence step-by-step)
    └── Attention (Computes alignment weights over Encoder hidden states)
"""
import random

import numpy as np
import torch
import yaml
from torch import nn

from src.features import get_windows
from src.noise import add_weather_noise


class Attention(nn.Module):
    """Scores how relevant each input hour is for the current decoder step."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)  # [decoder state; encoder output]
        self.V = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, dec_hidden, enc_outputs):
        # dec_hidden: [batch, layers, hidden], enc_outputs: [batch, input_hours, 2*hidden]
        s = dec_hidden[:, -1:, :].expand(-1, enc_outputs.size(1), -1)   # top layer, repeated per input hour
        energy = torch.tanh(self.W(torch.cat((s, enc_outputs), dim=2)))
        return torch.softmax(energy @ self.V, dim=1)                     # [batch, input_hours]


class Encoder(nn.Module):
    """Reads the input hours forwards and backwards."""

    def __init__(self, input_dim, n_layers, hidden_dim, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outputs, hidden = self.gru(x)                       # hidden: [2*layers, batch, hidden]
        batch = x.size(0)
        # Add forward + backward state of each layer -> start state of the decoder
        hidden = hidden.view(self.n_layers, 2, batch, -1).sum(dim=1).permute(1, 0, 2)
        return self.dropout(outputs), torch.sigmoid(self.fc(hidden))   # [batch, layers, hidden]


class Decoder(nn.Module):
    """Makes one forecast hour at a time."""

    def __init__(self, n_quantiles, future_dim, n_layers, hidden_dim, dropout):
        super().__init__()
        self.attention = Attention(hidden_dim)
        step_input = 1 + future_dim + 2 * hidden_dim     # [previous demand; this hour's time + weather; context]
        self.gru = nn.GRU(step_input, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(step_input + hidden_dim, n_quantiles)

    def forward(self, prev_y, future_t, hidden, enc_outputs):
        # prev_y: demand of the previous hour [batch]
        # future_t: time and weather of the hour being forecast [batch, future_dim]
        weights = self.attention(hidden, enc_outputs).unsqueeze(1)          # [batch, 1, input_hours]
        context = torch.bmm(weights, enc_outputs)[:, 0]                     # [batch, 2*hidden]
        step_in = torch.cat((prev_y[:, None], future_t, context), dim=1)
        out, hidden = self.gru(step_in[:, None], hidden.permute(1, 0, 2).contiguous())
        pred = self.fc(torch.cat((step_in, out[:, 0]), dim=1))
        return pred, hidden.permute(1, 0, 2)                                # [batch, n_quantiles]


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, future_dim, output_hours, quantiles, n_layers, hidden_dim, dropout):
        super().__init__()
        self.output_hours = output_hours
        self.median = quantiles.index(0.5)
        self.encoder = Encoder(input_dim, n_layers, hidden_dim, dropout)
        self.decoder = Decoder(len(quantiles), future_dim, n_layers, hidden_dim, dropout)

    def forward(self, past, last_y, future, targets=None, teacher_forcing=0.0):
        """past: features of the input hours, last_y: last known demand,
        future: time + weather of the forecast hours, targets: true future demand (training only)."""
        enc_outputs, hidden = self.encoder(past)
        prev_y = last_y
        preds = []
        for t in range(self.output_hours):
            pred, hidden = self.decoder(prev_y, future[:, t], hidden, enc_outputs)
            preds.append(pred)
            # Next step gets this hour's demand: the true value (teacher forcing) or the model's median
            if targets is not None and random.random() < teacher_forcing:
                prev_y = targets[:, t]
            else:
                prev_y = pred[:, self.median]
        return torch.stack(preds, dim=1)                                    # [batch, output_hours, n_quantiles]


def quantile_loss(preds, targets, quantiles):
    """Pinball loss for each quantile. preds: [batch, hours, n_q], targets: [batch, hours]."""
    q = torch.tensor(quantiles, device=preds.device)
    errors = targets.unsqueeze(2) - preds
    per_quantile = torch.maximum(q * errors, (q - 1) * errors).mean(dim=(0, 1))
    return per_quantile.sum(), per_quantile.detach().cpu().numpy()


def build_model(cfg, data):
    return Seq2Seq(data["X"].shape[1], data["X_future"].shape[1], cfg["output_hours"], cfg["quantiles"],
                   cfg["n_layers"], cfg["hidden_dim"], cfg["dropout"])


def run_folder_name(cfg):
    """e.g. T48-L3-HD64-perfect or T48-L3-HD64-noisy"""
    return f"{cfg['run_name']}-{cfg['future_weather']}"


def load_trained_model(cfg, data):
    """Load a model saved by 3_train.py, together with the settings it was trained with."""
    folder = f"models/{cfg['location']}/{run_folder_name(cfg)}"
    with open(f"{folder}/config.yaml", encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)
    model = build_model(run_cfg, data)
    model.load_state_dict(torch.load(f"{folder}/model.pt", map_location="cpu"))
    model.eval()
    return model, run_cfg


def forecast(model, data, starts, cfg, noisy_weather=False, seed=0, batch_size=256):
    """Forecast from every start hour without teacher forcing. Returns predictions and targets (scaled).

    noisy_weather=True gives the model forecast-like weather (see src/noise.py) instead of the observed one.
    The same seed gives the same noise, so results can be repeated and compared.
    """
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            past, last_y, future, target = get_windows(data, starts[i:i + batch_size], cfg)
            if noisy_weather:
                future = add_weather_noise(future, data, cfg, rng)
            past, last_y, future = [torch.from_numpy(a).to(device) for a in (past, last_y, future)]
            preds.append(model(past, last_y, future).cpu().numpy())
            targets.append(target)
    return np.concatenate(preds), np.concatenate(targets)

