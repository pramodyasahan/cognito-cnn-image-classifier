import torch

from utils.dataset_loader import load_mnist_torch
from models.cnn import CustomCNN
from utils.train_utils import train_model
from utils.metrics_utils import evaluate_and_save
from utils.plot_utils import plot_loss_curve

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) data
    train_loader, val_loader, test_loader = load_mnist_torch(batch_size=64, seed=42)

    # 2) model (baseline)
    def model_factory():
        return CustomCNN(
            in_channels=1,
            num_classes=10,
            x1=32, m1=3,
            x2=64, m2=3,
            fc_units=128,
            dropout=0.3,
            activation="relu",
        )
    model = model_factory()

    # 3) train 20 epochs (as required)
    history, ckpt_path = train_model(
        model,
        train_loader, val_loader,
        epochs=20,
        optimizer_name="adam",   # we will add SGD/SGDM runs later
        lr=1e-3,
        momentum=0.9,
        device=device,
        out_dir="reports"
    )

    # 4) plots + test metrics
    plot_loss_curve(history, save_path="reports/figures/loss_curve_custom.png")
    evaluate_and_save(
        model_class=model_factory,
        ckpt_path=ckpt_path,
        test_loader=test_loader,
        device=device,
        report_csv="reports/results.csv",
        cm_png="reports/figures/cm_custom.png"
    )

if __name__ == "__main__":
    main()
