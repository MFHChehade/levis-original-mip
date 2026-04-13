
import torch
import torch.nn as nn


class SequentialClassifier(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def build_model(model_name: str) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "mnist_mlp":
        return SequentialClassifier([
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ])

    if model_name == "mnist_cnn":
        return SequentialClassifier([
            nn.Conv2d(1, 4, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        ])

    if model_name == "cifar10_mlp":
        return SequentialClassifier([
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ])

    if model_name == "cifar10_cnn":
        return SequentialClassifier([
            nn.Conv2d(3, 4, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        ])

    raise ValueError(f"Unknown model_name: {model_name}")


def input_shape_for_model(model_name: str):
    model_name = model_name.lower()
    if model_name.startswith("mnist"):
        return (1, 28, 28)
    if model_name.startswith("cifar10"):
        return (3, 32, 32)
    raise ValueError(f"Unknown model_name: {model_name}")


def dataset_name_for_model(model_name: str) -> str:
    model_name = model_name.lower()
    if model_name.startswith("mnist"):
        return "mnist"
    if model_name.startswith("cifar10"):
        return "cifar10"
    raise ValueError(f"Unknown model_name: {model_name}")


def count_total_relu_units(model: nn.Module) -> int:
    current_shape = None
    total = 0
    for module in model.layers:
        if isinstance(module, nn.Flatten):
            if current_shape is None:
                raise RuntimeError("Input shape must be tracked before Flatten.")
            current_shape = (int(torch.tensor(current_shape).prod().item()),)
        elif isinstance(module, nn.Linear):
            current_shape = (module.out_features,)
        elif isinstance(module, nn.Conv2d):
            if current_shape is None:
                raise RuntimeError("Input shape must be tracked before Conv2d.")
            c_in, h_in, w_in = current_shape
            kh, kw = module.kernel_size
            sh, sw = module.stride
            ph, pw = module.padding
            dh, dw = module.dilation
            h_out = ((h_in + 2 * ph - dh * (kh - 1) - 1) // sh) + 1
            w_out = ((w_in + 2 * pw - dw * (kw - 1) - 1) // sw) + 1
            current_shape = (module.out_channels, h_out, w_out)
        elif isinstance(module, nn.ReLU):
            if current_shape is None:
                raise RuntimeError("Input shape must be tracked before ReLU.")
            total += int(torch.tensor(current_shape).prod().item())
        else:
            raise TypeError(f"Unsupported module type: {type(module)}")
    return total

