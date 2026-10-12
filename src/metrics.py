"""Forecast scores in kW. y: [n_days, hours], preds: [n_days, hours, n_quantiles]."""
import numpy as np


def compute_metrics(y, preds, quantiles):
    low, mid, high = 0, quantiles.index(0.5), len(quantiles) - 1
    median = preds[:, :, mid]
    m = {}
    for i, q in enumerate(quantiles):
        error = y - preds[:, :, i]
        m[f"pinball_q{q}"] = float(np.mean(np.maximum(q * error, (q - 1) * error)))
    m["mae"] = float(np.mean(np.abs(median - y)))
    m["rmse"] = float(np.sqrt(np.mean((median - y) ** 2)))
    m["mape_%"] = float(100 * np.mean(np.abs(median - y) / np.abs(y)))

    # How often the actual value falls inside the q0.1-q0.9 band (should be ~80%)
    inside = (y >= preds[:, :, low]) & (y <= preds[:, :, high])
    m["interval_coverage"] = float(inside.mean())
    m["interval_expected"] = quantiles[high] - quantiles[low]
    m["interval_width_kw"] = float(np.mean(preds[:, :, high] - preds[:, :, low]))

    # Share of actual values below each quantile forecast (should be close to the quantile itself)
    for i, q in enumerate(quantiles):
        m[f"share_below_q{q}"] = float(np.mean(y <= preds[:, :, i]))
    return m
