"""Latih CNN dari awal: python train.py --dataset-root ... --mode butterworth"""
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from src.preprocessing import load_lab_windows, split_scale
from src.models import CNNOnly

def loader(x, y, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.tensor(y, dtype=torch.long)),
                      batch_size=32, shuffle=shuffle)

def evaluate(model, data_loader, device, criterion):
    model.eval()
    losses, ys, preds = [], [], []
    with torch.no_grad():
        for x, y in data_loader:
            logits = model(x.to(device))
            losses.append((criterion(logits, y.to(device)).item(), len(y)))
            ys.extend(y.numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
    loss = sum(v * n for v, n in losses) / sum(n for _, n in losses)
    return loss, accuracy_score(ys, preds), np.asarray(ys), np.asarray(preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--mode", choices=["combined", "butterworth"], default="butterworth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", default="results/run")
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    X, labels, groups = load_lab_windows(args.dataset_root, args.mode)
    splits, encoder, scaler = split_scale(X, labels, groups)
    (xtr, ytr), (xva, yva), (xte, yte) = splits
    train_loader = loader(xtr, ytr, True)
    val_loader = loader(xva, yva)
    test_loader = loader(xte, yte)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNOnly(xtr.shape[-1], len(encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_state, best_epoch = [], float("inf"), None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        train_loss, train_acc, _, _ = evaluate(model, loader(xtr, ytr), device, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch:02d} | train={train_acc:.4f} | val={val_acc:.4f} | val_loss={val_loss:.4f}")
    model.load_state_dict(best_state)
    _, acc, true, pred = evaluate(model, test_loader, device, criterion)
    f1 = f1_score(true, pred, average="macro", zero_division=0)
    torch.save(dict(model_state_dict=best_state, classes=encoder.classes_.tolist(),
                    mode=args.mode, in_features=xtr.shape[-1],
                    best_epoch=best_epoch), out / "checkpoint.pth")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    pd.DataFrame(dict(actual_id=encoder.inverse_transform(true),
                      predicted_id=encoder.inverse_transform(pred))).to_csv(
                          out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(dict(test_accuracy=acc, macro_f1=f1,
                                    best_epoch=best_epoch, best_val_loss=best_loss), indent=2))
    print(f"TEST accuracy={acc:.4%} | macro F1={f1:.4f}")

if __name__ == "__main__":
    main()
