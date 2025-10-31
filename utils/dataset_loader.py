from torch.utils.data import Dataset

class _TensorDatasetWithTransform(Dataset):
    def __init__(self, X, y, transform):
        self.X = X  # [N,1,28,28] float32 in [0,1]
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        img = self.X[idx]               # tensor [1,28,28]
        label = int(self.y[idx])
        img_hwc = img.permute(1, 2, 0).numpy()  # [28,28,1]
        out = self.transform(img_hwc)
        return out, label


def load_mnist_torch_rgb224(batch_size=64, seed=42, num_workers=2):
    """
    Same split as load_mnist_torch(), but outputs 224x224, 3-channel tensors
    using a simple transform: repeat channel + resize + normalize.
    """
    import numpy as np
    import torch
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    import os

    # base dataset
    base = transforms.ToTensor()
    d_train = datasets.MNIST(root="data", train=True, download=True, transform=base)
    d_test = datasets.MNIST(root="data", train=False, download=True, transform=base)

    X_train = d_train.data.unsqueeze(1).float() / 255.0
    y_train = d_train.targets.numpy()
    X_test = d_test.data.unsqueeze(1).float() / 255.0
    y_test = d_test.targets.numpy()
    X = torch.cat([X_train, X_test], dim=0)
    y = np.concatenate([y_train, y_test], axis=0)

    SPLIT_DIR = "data/splits"
    tr = np.load(os.path.join(SPLIT_DIR, "train.npy"))
    va = np.load(os.path.join(SPLIT_DIR, "val.npy"))
    te = np.load(os.path.join(SPLIT_DIR, "test.npy"))

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    ds_train = _TensorDatasetWithTransform(X[tr], y[tr], tfm)
    ds_val = _TensorDatasetWithTransform(X[va], y[va], tfm)
    ds_test = _TensorDatasetWithTransform(X[te], y[te], tfm)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
