from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import get_datasets, maybe_subset
from .models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        preds = logits.argmax(dim=1)

        total += yb.numel()
        correct += (preds == yb).sum().item()
        loss_sum += loss.item() * yb.size(0)

    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "n": total,
    }


def resolve_device(device_name: str) -> torch.device:
    requested = str(device_name).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mnist_mlp", "mnist_cnn", "cifar10_mlp", "cifar10_cnn"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_subset", type=int, default=0)
    parser.add_argument("--test_subset", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = get_datasets(args.model, args.data_dir)
    train_ds = maybe_subset(train_ds, args.train_subset if args.train_subset > 0 else None)
    test_ds = maybe_subset(test_ds, args.test_subset if args.test_subset > 0 else None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = resolve_device(args.device)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total += yb.numel()
            correct += (preds == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)

        train_metrics = {
            "loss": loss_sum / max(total, 1),
            "acc": correct / max(total, 1),
            "n": total,
        }
        test_metrics = evaluate(model, test_loader, device)

        row = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        history.append(row)
        print(json.dumps(row))

        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            best_state = {
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "best_test_acc": best_acc,
                "history": history,
                "seed": args.seed,
            }

    ckpt_path = out_dir / f"{args.model}.pt"
    torch.save(best_state, ckpt_path)

    summary = {
        "model": args.model,
        "checkpoint": str(ckpt_path),
        "best_test_acc": best_acc,
        "epochs": args.epochs,
        "train_subset": args.train_subset,
        "test_subset": args.test_subset,
        "device": str(device),
    }
    (out_dir / f"{args.model}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
