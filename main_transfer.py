import torch
from utils.dataset_loader import load_mnist_torch_rgb224
from models.transfer_models import get_resnet18, get_vgg16
from utils.transfer_train import train_transfer, freeze_backbone
from utils.metrics_utils import evaluate_and_save
from utils.plot_utils import plot_loss_curve

def run_one(model_fn, tag, device):
    # data
    train_loader, val_loader, test_loader = load_mnist_torch_rgb224(batch_size=64, seed=42)

    # model
    model = model_fn()
    model = freeze_backbone(model)

    # train
    history, ckpt_path = train_transfer(
        model,
        train_loader, val_loader,
        epochs=12,        # 10–15 is fine; keep it practical
        lr=1e-3,
        device=device,
        out_dir="reports",
        tag=tag
    )

    # plots + test eval
    plot_loss_curve(history, save_path=f"reports/figures/loss_curve_{tag}.png")

    # we need a factory that rebuilds the same architecture to load weights
    def factory():
        return model_fn()

    evaluate_and_save(
        model_class=factory,
        ckpt_path=ckpt_path,
        test_loader=test_loader,
        device=device,
        report_csv="reports/final_results.csv",
        cm_png=f"reports/figures/cm_{tag}.png"
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # run both backbones
    run_one(lambda: get_resnet18(num_classes=10, pretrained=True), tag="resnet18", device=device)
    run_one(lambda: get_vgg16(num_classes=10, pretrained=True),    tag="vgg16",    device=device)

    print("\ntransfer learning done. see `reports/figures/` and `reports/final_results.csv`.")

if __name__ == "__main__":
    main()
