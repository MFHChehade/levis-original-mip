
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .models import build_model, dataset_name_for_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_datasets(model_name: str, data_dir: Path):
    transform = transforms.ToTensor()
    dataset_name = dataset_name_for_model(model_name)

    if dataset_name == "mnist":
        train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        return train_ds, test_ds

    if dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return train_ds, test_ds

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def maybe_subset(dataset, subset_size: int | None):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    indices = list(range(subset_size))
    return Subset(dataset, indices)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
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

    train_ds, test_ds = get_datasets(args.model, Path(args.data_dir))
    train_ds = maybe_subset(train_ds, args.train_subset if args.train_subset > 0 else None)
    test_ds = maybe_subset(test_ds, args.test_subset if args.test_subset > 0 else None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
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

