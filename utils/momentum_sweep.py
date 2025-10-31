import os
import torch
import matplotlib.pyplot as plt

from models.cnn import CustomCNN
from utils.train_utils import train_model
from utils.dataset_loader import load_mnist_torch


def run_momentum_sweep(
    momentum_list=(0.0, 0.5, 0.9),
    lr=1e-2,
    epochs=10,
    batch_size=64,
    device=None,
    fig_path="reports/figures/momentum_effect.png",
):
    # pick device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ensure output folder exists
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    # same split reused for all runs
    train_loader, val_loader, _ = load_mnist_torch(batch_size=batch_size, seed=42)

    val_accs = []
    for m in momentum_list:
        print(f"\n=== running SGD with momentum={m} ===")
        model = CustomCNN()
        history, _ = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            optimizer_name="sgdm",  # our train_utils maps this to SGD with momentum
            lr=lr,
            momentum=m,
            device=device,
            out_dir="reports",
        )
        # take the last recorded validation accuracy
        val_accs.append(history["val_acc"][-1])

    # plot the relationship
    plt.figure()
    plt.plot(list(momentum_list), val_accs, marker="o")
    plt.title("Effect of Momentum on Validation Accuracy")
    plt.xlabel("Momentum")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"\nMomentum sweep done. Saved plot to {fig_path}")


if __name__ == "__main__":
    # sensible defaults; tweak if needed
    run_momentum_sweep(
        momentum_list=(0.0, 0.5, 0.9),
        lr=1e-2,
        epochs=10,       # short run just for the trend
        batch_size=64,
        device=None,     # auto-pick CUDA if available
        fig_path="reports/figures/momentum_effect.png",
    )
