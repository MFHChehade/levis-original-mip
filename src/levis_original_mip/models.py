from __future__ import annotations

import torch.nn as nn


class SmallSequentialNet(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def dataset_name_for_model(model_name: str) -> str:
    if model_name.startswith("mnist_"):
        return "mnist"
    if model_name.startswith("cifar10_"):
        return "cifar10"
    raise ValueError(f"Unsupported model name: {model_name}")


def build_model(model_name: str) -> nn.Module:
    if model_name == "mnist_mlp":
        return SmallSequentialNet(
            [
                nn.Flatten(),
                nn.Linear(28 * 28, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            ]
        )

    if model_name == "mnist_cnn":
        return SmallSequentialNet(
            [
                nn.Conv2d(1, 4, kernel_size=4, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(4 * 7 * 7, 32),
                nn.ReLU(),
                nn.Linear(32, 10),
            ]
        )

    if model_name == "cifar10_mlp":
        return SmallSequentialNet(
            [
                nn.Flatten(),
                nn.Linear(3 * 32 * 32, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            ]
        )

    if model_name == "cifar10_cnn":
        return SmallSequentialNet(
            [
                nn.Conv2d(3, 4, kernel_size=4, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(4 * 8 * 8, 32),
                nn.ReLU(),
                nn.Linear(32, 10),
            ]
        )

    raise ValueError(f"Unsupported model name: {model_name}")
