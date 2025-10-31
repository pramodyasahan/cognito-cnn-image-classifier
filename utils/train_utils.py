import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def get_optimizer(name, params, lr=1e-3, momentum=0.9):
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.0)
    if name in ("sgdm", "sgd_momentum"):
        return optim.SGD(params, lr=lr, momentum=momentum)
    return optim.Adam(params, lr=lr)  # default

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=20,
    optimizer_name="adam",
    lr=1e-3,
    momentum=0.9,
    device="cpu",
    out_dir="reports",
):
    os.makedirs(out_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr, momentum=momentum)

    model.to(device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = os.path.join(out_dir, f"{model.__class__.__name__.lower()}_{optimizer_name}.pt")

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        # ---- val ----
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_loss = total_loss / total
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

    return history, best_path
