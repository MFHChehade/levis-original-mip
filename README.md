# Original LEVIS MIP: Exact ReLU Verification for MNIST and CIFAR-10

This repository implements the **original exact mixed-integer formulation** behind the LEVIS-style nearest-adversarial-point problem for small ReLU networks on **MNIST** and **CIFAR-10**.

The project currently supports four model families:

- `mnist_mlp`
- `mnist_cnn`
- `cifar10_mlp`
- `cifar10_cnn`

It also supports two solver modes:

- **CBC** for lightweight smoke tests
- **Gurobi** for fuller exact runs

## What the code does

Given a trained ReLU classifier and a correctly classified test input, the code builds an **exact targeted** mixed-integer model for the nearest adversarial example under an $\ell_\infty$ objective.

At a high level, the optimization problem is:

$$
\min_x \|x - x_0\|_\infty
$$

subject to:

- exact affine layer equations
- exact ReLU big-M constraints
- input bounds
- targeted misclassification margin constraint

## Current status

The repository is set up to:

1. train the four small networks,
2. save checkpoints,
3. run end-to-end smoke tests,
4. solve exact MIPs with either CBC or Gurobi.

CBC smoke tests have already been validated on all four models in a local run.

## Repository layout

```text
levis-original-mip/
+-- checkpoints/              # generated locally, not committed
+-- data/                     # generated locally, not committed
+-- smoke_results/            # generated locally, usually not committed
+-- exact_mip.py
+-- run_smoke_tests.py
+-- train_classifier.py
+-- README.md
+-- requirements.txt
+-- .gitignore
```

## Installation

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training

Example smoke-training commands:

```powershell
python train_classifier.py --model mnist_mlp   --out_dir checkpoints --data_dir data --device cpu --epochs 3 --batch_size 128 --lr 1e-3 --train_subset 12000 --test_subset 2000
python train_classifier.py --model mnist_cnn   --out_dir checkpoints --data_dir data --device cpu --epochs 5 --batch_size 128 --lr 1e-3 --train_subset 12000 --test_subset 2000
python train_classifier.py --model cifar10_mlp --out_dir checkpoints --data_dir data --device cpu --epochs 6 --batch_size 128 --lr 1e-3 --train_subset 20000 --test_subset 4000
python train_classifier.py --model cifar10_cnn --out_dir checkpoints --data_dir data --device cpu --epochs 8 --batch_size 128 --lr 1e-3 --train_subset 20000 --test_subset 4000
```

## Smoke tests with CBC

```powershell
python run_smoke_tests.py --root_dir . --data_dir data --time_limit 15 --mip_gap 0.05 --max_candidates 1000 --solver cbc
```

## Full exact runs with Gurobi

```powershell
python exact_mip.py --checkpoint checkpoints\mnist_mlp.pt   --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json smoke_results\mnist_mlp_full_gurobi.json
python exact_mip.py --checkpoint checkpoints\mnist_cnn.pt   --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json smoke_results\mnist_cnn_full_gurobi.json
python exact_mip.py --checkpoint checkpoints\cifar10_mlp.pt --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json smoke_results\cifar10_mlp_full_gurobi.json
python exact_mip.py --checkpoint checkpoints\cifar10_cnn.pt --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json smoke_results\cifar10_cnn_full_gurobi.json
```

## Notes

- CBC is intended for portability and smoke testing.
- Gurobi is the recommended solver for more serious exact runs.
- The current code targets small, MIP-friendly ReLU architectures.
- Large CNNs are in principle MIP-encodable, but exact solving becomes much harder.

## Suggested GitHub title

**Original LEVIS MIP: Exact ReLU Verification for MNIST and CIFAR-10**

## Suggested short repository name

`levis-original-mip`
