# tools/plot_task1_log.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # 项目根目录 = 当前文件的上一级
    project_root = Path(__file__).resolve().parent.parent
    log_path = project_root / "outputs" / "task1_train_log.csv"
    save_dir = project_root / "outputs" / "vis_task1"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "task1_curves.png"

    print(f"[Load] reading log from: {log_path}")
    df = pd.read_csv(log_path)

    # 取出各列
    epochs = df["epoch"]
    train_loss = df["train_loss"].astype(float)
    train_acc = df["train_acc"].astype(float)
    val_loss = df["val_loss"].astype(float)
    val_acc = df["val_acc"].astype(float)
    best_val_acc = df["best_val_acc"].astype(float)

    # 找到 val_acc 最优的 epoch
    best_idx = val_acc.argmax()
    best_epoch = int(epochs.iloc[best_idx])
    best_acc_value = float(val_acc.iloc[best_idx])
    print(f"[Info] best val_acc = {best_acc_value:.4f} at epoch {best_epoch}")

    # Create a figure with two subplots (vertically stacked)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # ---------- Loss subplot ----------
    ax1.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=4)
    ax1.plot(epochs, val_loss, label="Val Loss", marker="s", markersize=4)
    ax1.axvline(best_epoch, linestyle="--", alpha=0.5, label=f"Best Epoch {best_epoch}")
    ax1.set_ylabel("Loss")
    ax1.set_title("Task1 Training / Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---------- Accuracy subplot ----------
    ax2.plot(epochs, train_acc, label="Train Acc", marker="o", markersize=4)
    ax2.plot(epochs, val_acc, label="Val Acc", marker="s", markersize=4)
    ax2.plot(epochs, best_val_acc, label="Best Val Acc (running)", linestyle=":")
    ax2.axvline(best_epoch, linestyle="--", alpha=0.5, label=f"Best Epoch {best_epoch}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Task1 Training / Validation Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)