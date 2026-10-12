"""Figures saved as PNG files."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
BLACK, GREY = "#0b0b0b", "#52514e"


def tidy(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df")


def plot_training(history, quantiles, file):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")

    axs[0, 0].plot(epochs, history["train_loss"], color=BLUE, label="Train")
    axs[0, 0].plot(epochs, history["val_loss"], color=ORANGE, label="Validation")
    axs[0, 0].legend(frameon=False)
    tidy(axs[0, 0], "Quantile loss", "Epoch")

    val_q = np.array(history["val_loss_per_quantile"])
    for i, (q, color) in enumerate(zip(quantiles, [BLUE, ORANGE, GREEN])):
        axs[0, 1].plot(epochs, val_q[:, i], color=color, label=f"q = {q}")
    axs[0, 1].legend(frameon=False)
    tidy(axs[0, 1], "Validation loss per quantile", "Epoch")

    axs[1, 0].plot(epochs, history["val_mae"], color=BLUE)
    tidy(axs[1, 0], "Validation MAE of the median (kW)", "Epoch")

    axs[1, 1].plot(epochs, history["val_coverage"], color=BLUE)
    expected = quantiles[-1] - quantiles[0]
    axs[1, 1].axhline(expected, color=GREY, linestyle="--", linewidth=1)
    axs[1, 1].text(1, expected + 0.02, f"expected {expected:.0%}", color=GREY)
    axs[1, 1].set_ylim(0, 1)
    tidy(axs[1, 1], f"Validation coverage of the q{quantiles[0]}-q{quantiles[-1]} band", "Epoch")

    fig.savefig(file, dpi=200)
    plt.close(fig)


def plot_forecast(table, quantiles, title, file):
    """table: timestamp, one column per quantile, and optionally actual_kw."""
    low, mid, high = (f"q{quantiles[0]}_kw", f"q{quantiles[len(quantiles) // 2]}_kw", f"q{quantiles[-1]}_kw")
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.fill_between(table["timestamp"], table[low], table[high], color=BLUE, alpha=0.18, linewidth=0,
                    label=f"q{quantiles[0]}-q{quantiles[-1]} band")
    ax.plot(table["timestamp"], table[mid], color=BLUE, linewidth=2, label="Median forecast")
    if "actual_kw" in table:
        ax.plot(table["timestamp"], table["actual_kw"], color=BLACK, linewidth=1.2, label="Actual")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    tidy(ax, title, ylabel="Demand per household (kW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.savefig(file, dpi=200)
    plt.close(fig)


def plot_calibration(metrics, quantiles, file):
    shares = [metrics[f"share_below_q{q}"] for q in quantiles]
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1)
    ax.plot(quantiles, shares, color=BLUE, marker="o", markersize=8, linewidth=2)
    for q, s in zip(quantiles, shares):
        ax.annotate(f"{s:.2f}", (q, s), xytext=(-8, 8), textcoords="offset points", ha="right", color=GREY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    tidy(ax, "Quantile calibration", "Quantile", "Share of actual values below forecast")
    fig.savefig(file, dpi=200)
    plt.close(fig)


def plot_error_by_hour_ahead(mae_per_hour, file):
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    ax.plot(np.arange(1, len(mae_per_hour) + 1), mae_per_hour, color=BLUE, marker="o", markersize=4, linewidth=2)
    ax.set_ylim(bottom=0)
    tidy(ax, "Error of the median by hours ahead", "Hours ahead", "MAE (kW)")
    fig.savefig(file, dpi=200)
    plt.close(fig)


def sample_week(forecasts, start):
    """Rows of the forecasts table for the 7 days starting at `start`."""
    start = pd.Timestamp(start)
    return forecasts[(forecasts["timestamp"] >= start) & (forecasts["timestamp"] < start + pd.Timedelta(days=7))]
