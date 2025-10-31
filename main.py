import torch

from utils.dataset_loader import load_mnist_torch
from models.cnn import CustomCNN
from utils.train_utils import train_model
from utils.metrics_utils import evaluate_and_save
from utils.plot_utils import plot_loss_curve

def make_model():
    # small, clean CNN as per the assignment
    return CustomCNN(
        in_channels=1,
        num_classes=10,
        x1=32, m1=3,
        x2=64, m2=3,
        fc_units=128,
        dropout=0.3,
        activation="relu",
    )

def run_one(optimizer_name: str, lr: float, momentum: float, device: str):
    # fresh model every time
    model = make_model()

    # train 20 epochs (assignment requirement)
    history, ckpt_path = train_model(
        model,
        train_loader, val_loader,
        epochs=20,
        optimizer_name=optimizer_name,
        lr=lr,
        momentum=momentum,
        device=device,
        out_dir="reports"
    )

    # filenames tagged by optimizer
    loss_png = f"reports/figures/loss_curve_{optimizer_name}.png"
    cm_png   = f"reports/figures/cm_{optimizer_name}.png"

    # save curve
    plot_loss_curve(history, save_path=loss_png)

    # test metrics + confusion matrix image
    evaluate_and_save(
        model_class=make_model,   # rebuild model to load the best checkpoint
        ckpt_path=ckpt_path,
        test_loader=test_loader,
        device=device,
        report_csv="reports/results.csv",
        cm_png=cm_png
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # 1) data (same split reused for all runs)
    global train_loader, val_loader, test_loader
    train_loader, val_loader, test_loader = load_mnist_torch(batch_size=64, seed=42)

    # 2) optimizer runs
    # note: slightly larger LR for (plain) SGD is common; momentum run uses same LR
    runs = [
        ("adam", 1e-3, 0.0),   # Adam
        ("sgd",  1e-2, 0.0),   # SGD
        ("sgdm", 1e-2, 0.9),   # SGD + Momentum=0.9
    ]
    for name, lr, mom in runs:
        print(f"\n=== running {name} (lr={lr}, momentum={mom}) ===")
        run_one(name, lr, mom, device)

    print("\nSuccessfully completed")

if __name__ == "__main__":
    main()
