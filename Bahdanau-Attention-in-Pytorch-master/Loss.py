import numpy as np
import torch
from torch import nn


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        if not all(0 < q < 1 for q in quantiles):
            raise ValueError("All quantiles must be between 0 and 1.")
        self.quantiles = torch.tensor(quantiles)

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predicted values, shape [batch_size, seq_len, num_quantiles].
            targets (torch.Tensor): True values, shape [batch_size, seq_len, num_quantiles].
        Returns:
            torch.Tensor: Combined quantile loss.
        """
        try:
            preds.shape[2] == len(self.quantiles)
        except ValueError:
            print(f'Output_dim of predictions must equal {len(self.quantiles)}, got {preds.shape[2]} instead')
        try:
            targets.shape[2] == len(self.quantiles)
        except ValueError:
            print(f'Output_dim of targets must equal {len(self.quantiles)}, got {targets.shape[2]} instead')

        errors = targets - preds  # [batch_size, seq_len, output_dim]
        losses = [torch.maximum(q * errors[:, :, i], (q - 1) * errors[:, :, i]).mean()
                  for i, q in enumerate(self.quantiles)]
        total_loss = torch.stack(losses).sum()  # scalar
        # loss for each quantile
        with torch.no_grad():
            quantile_losses = torch.stack(losses).cpu().numpy()

        return total_loss, quantile_losses
