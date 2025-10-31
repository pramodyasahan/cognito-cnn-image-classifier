import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def evaluate_and_save(model_class, ckpt_path, test_loader, device="cpu",
                      report_csv="reports/results.csv",
                      cm_png="reports/figures/cm_custom.png"):
    os.makedirs(os.path.dirname(report_csv), exist_ok=True)
    os.makedirs(os.path.dirname(cm_png), exist_ok=True)

    model = model_class()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(list(preds))
            y_true.extend(list(yb.numpy()))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # write csv (append)
    header = ["checkpoint", "accuracy", "precision_macro", "recall_macro"]
    row = [ckpt_path, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}"]
    try:
        with open(report_csv, "x", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerow(row)
    except FileExistsError:
        with open(report_csv, "a", newline="") as f:
            w = csv.writer(f); w.writerow(row)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix – Test Set")
    plt.tight_layout()
    plt.savefig(cm_png, dpi=150)
    plt.close()

    print(f"test accuracy: {acc:.4f} | precision_macro: {prec:.4f} | recall_macro: {rec:.4f}")
