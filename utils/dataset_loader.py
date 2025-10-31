import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms

SPLIT_DIR = "data/splits"
os.makedirs(SPLIT_DIR, exist_ok=True)

def _save_idx(name, idx):
    np.save(os.path.join(SPLIT_DIR, f"{name}.npy"), idx)

def _load_idx(name):
    path = os.path.join(SPLIT_DIR, f"{name}.npy")
    return np.load(path) if os.path.exists(path) else None

def load_mnist_torch(batch_size=64, seed=42, num_workers=2):
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),  # [0,1], shape [1,28,28]
    ])

    # download train and test parts then combine (we will re-split 70/15/15)
    d_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    d_test  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    X_train = d_train.data.unsqueeze(1).float() / 255.0
    y_train = d_train.targets.numpy()
    X_test  = d_test.data.unsqueeze(1).float() / 255.0
    y_test  = d_test.targets.numpy()

    X = torch.cat([X_train, X_test], dim=0)       # [N,1,28,28]
    y = np.concatenate([y_train, y_test], axis=0) # [N]

    # try to load saved indices
    tr_idx = _load_idx("train")
    va_idx = _load_idx("val")
    te_idx = _load_idx("test")

    if tr_idx is None or va_idx is None or te_idx is None:
        # 70% train vs 30% temp
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        tr, temp = next(sss1.split(np.zeros_like(y), y))
        # from temp (30%), split into 15% val and 15% test
        y_temp = y[temp]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        va_rel, te_rel = next(sss2.split(np.zeros_like(y_temp), y_temp))
        va, te = temp[va_rel], temp[te_rel]

        tr_idx, va_idx, te_idx = tr, va, te
        _save_idx("train", tr_idx)
        _save_idx("val",   va_idx)
        _save_idx("test",  te_idx)

    # build TensorDatasets from the split
    ds_train = TensorDataset(X[tr_idx], torch.from_numpy(y[tr_idx]).long())
    ds_val   = TensorDataset(X[va_idx], torch.from_numpy(y[va_idx]).long())
    ds_test  = TensorDataset(X[te_idx], torch.from_numpy(y[te_idx]).long())

    # data loaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
