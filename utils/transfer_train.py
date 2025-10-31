import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def freeze_backbone(model):
    # try common attributes; if layer names differ, that's fine — we only need params with requires_grad=False
    for name, param in model.named_parameters():
        param.requires_grad = False
    # then unfreeze the final classifier layers
    # resnet: 'fc'
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    # vgg: 'classifier'
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    return model

def train_transfer(
    model,
    train_loader,
    val_loader,
    epochs=12,
    lr=1e-3,
    device="cpu",
    out_dir="reports",
    tag="resnet18",
):
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = os.path.join(out_dir, f"{tag}_transfer.pt")

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in tqdm(train_loader, desc=f"{tag} epoch {epoch}/{epochs} [train]", leave=False):
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

        # val
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"{tag} epoch {epoch}/{epochs} [val]", leave=False):
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        print(f"{tag} | epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

    return history, best_path
