"""Step 3: train the model.

With future_weather: noisy, the weather of the forecast hours gets new forecast-like errors in every
batch, so the model learns not to trust it fully.

Saves (<run> = <run_name>-perfect or <run_name>-noisy):
    models/<location>/<run>/model.pt        weights of the best epoch (lowest validation loss)
    models/<location>/<run>/config.yaml     the settings used, so results can be reproduced
    models/<location>/<run>/history.csv     loss and scores per epoch
    reports/<location>/<run>/training.png
"""
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch

from src.features import demand_to_kw, get_windows, load_features, window_starts
from src.metrics import compute_metrics
from src.model import build_model, forecast, quantile_loss, run_folder_name
from src.noise import add_weather_noise
from src.plots import plot_training
from src.utils import config_path, load_config

cfg = load_config()
run = run_folder_name(cfg)
noisy = cfg["future_weather"] == "noisy"
model_folder = f"models/{cfg['location']}/{run}"
report_folder = f"reports/{cfg['location']}/{run}"
os.makedirs(model_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)
shutil.copy(config_path(), f"{model_folder}/config.yaml")

random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])
rng = np.random.default_rng(cfg["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
data = load_features(cfg["location"])
scaler = data["scaler"]
train_starts = window_starts(data["df"], "train", cfg)
val_starts = window_starts(data["df"], "val", cfg, midnight_only=True)  # one day-ahead forecast per validation day

# Model
quantiles = cfg["quantiles"]
model = build_model(cfg, data).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=cfg["learning_rate"],
    amsgrad=True,
    weight_decay=cfg["weight_decay"])
print(f"Training {run} on {device}: {len(train_starts):,} training windows, {len(val_starts)} validation days")

history = {"train_loss": [], "val_loss": [], "val_loss_per_quantile": [], "val_mae": [], "val_coverage": []}
best_val_loss, epochs_without_improvement = float("inf"), 0

for epoch in range(cfg["n_epochs"]):
    start_time = time.time()

    # Learning rate and teacher forcing shrink by 1/sqrt(epoch) after the first two epochs
    decay = 1 if epoch < 2 else 1 / np.sqrt(epoch + 2)
    for group in optimizer.param_groups:
        group["lr"] = cfg["learning_rate"] * decay
    teacher_forcing = cfg["teacher_forcing"] * decay

    # Train on random windows
    model.train()
    train_losses = []
    for _ in range(cfg["batches_per_epoch"]):
        starts = rng.choice(train_starts, size=cfg["batch_size"], replace=False)
        past, last_y, future, target = get_windows(data, starts, cfg)
        if noisy:
            future = add_weather_noise(future, data, cfg, rng)
        past, last_y, future, target = [torch.from_numpy(a).to(device) for a in (past, last_y, future, target)]
        optimizer.zero_grad()
        preds = model(past, last_y, future, target, teacher_forcing)
        loss, _ = quantile_loss(preds, target, quantiles)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        optimizer.step()
        
        train_losses.append(loss.item())

    # Validate: forecast every validation day without teacher forcing (same noise every epoch)
    val_preds, val_target = forecast(model, data, val_starts, cfg, noisy_weather=noisy, seed=cfg["seed"])
    val_loss, val_loss_per_q = quantile_loss(torch.from_numpy(val_preds), torch.from_numpy(val_target), quantiles)
    scores = compute_metrics(demand_to_kw(val_target, scaler), demand_to_kw(val_preds, scaler), quantiles)

    history["train_loss"].append(np.mean(train_losses))
    history["val_loss"].append(val_loss.item())
    history["val_loss_per_quantile"].append(val_loss_per_q)
    history["val_mae"].append(scores["mae"])
    history["val_coverage"].append(scores["interval_coverage"])
    print(f"Epoch {epoch + 1:3d}/{cfg['n_epochs']}  {time.time() - start_time:4.1f}s  "
          f"train loss {history['train_loss'][-1]:.4f}  val loss {val_loss.item():.4f}  "
          f"val MAE {scores['mae']:.3f} kW  val coverage {scores['interval_coverage']:.2f}")

    # Keep the best model; stop when validation loss has not improved for `patience` epochs
    if val_loss.item() < best_val_loss - cfg["min_delta"]:
        best_val_loss, epochs_without_improvement = val_loss.item(), 0
        torch.save(model.state_dict(), f"{model_folder}/model.pt")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= cfg["patience"]:
            print(f"Early stopping after {epoch + 1} epochs")
            break

table = pd.DataFrame({k: v for k, v in history.items() if k != "val_loss_per_quantile"})
for i, q in enumerate(quantiles):
    table[f"val_loss_q{q}"] = [row[i] for row in history["val_loss_per_quantile"]]
table.index.name = "epoch"
table.index += 1
table.to_csv(f"{model_folder}/history.csv")
plot_training(history, quantiles, f"{report_folder}/training.png")
print(f"Best validation loss {best_val_loss:.4f}. Model saved to {model_folder}/model.pt")
