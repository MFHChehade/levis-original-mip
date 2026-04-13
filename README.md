# Original LEVIS MIP: Exact ReLU Verification for MNIST and CIFAR-10

This repository implements a handoff-ready Python project for training small ReLU networks and solving the **original exact targeted \(\ell_\infty\)** mixed-integer formulation for nearest-adversarial-point search.

The code supports four model families:

- `mnist_mlp`
- `mnist_cnn`
- `cifar10_mlp`
- `cifar10_cnn`

The exact MIP supports two solver backends:

- **CBC** for smoke tests and portability
- **Gurobi** for full exact runs

## Repository layout

```text
src/levis_original_mip/
  __init__.py
  __main__.py
  data.py
  models.py
  training.py
  exact_mip.py
  run_smoke_tests.py

scripts/
  setup_env.ps1
  train_smoke.ps1
  train_full.ps1
  run_smoke_cbc.ps1
  run_full_gurobi.ps1
  clean_generated.ps1
  publish_to_github.ps1
```

## What this repository implements

### Training
The training module trains one of the supported MNIST or CIFAR-10 models and saves a checkpoint containing:

- `model_name`
- `state_dict`
- `best_test_acc`
- training history
- seed

### Exact MIP
The exact MIP module builds the original exact targeted \(\ell_\infty\) problem:

- decision variable: perturbed input \(x\)
- objective: minimize \(t\)
- perturbation constraints: \(|x - x_0| \le t\)
- exact affine layer equations for `Linear` and `Conv2d`
- exact Big-M ReLU constraints
- one binary per unstable ReLU
- targeted margin-violation constraint:
  `logit[true_class] - logit[target_class] <= 0`

### Smoke-test runner
The smoke-test runner searches for one correctly classified example per model and solves one exact MIP instance.

## Assumptions for the recipient

### Full runs
The intended full path assumes:
- a working **CUDA-capable PyTorch** install for training
- a working **Gurobi** installation and license for exact MIP runs

### Smoke path
The smoke path can run with:
- CPU PyTorch
- CBC via PuLP

## Recommended workflow

### 1. Set up the environment
```powershell
.\scripts\setup_env.ps1
```

### 2. Full training
```powershell
.\scripts\train_full.ps1
```

### 3. Full exact Gurobi runs
```powershell
.\scripts\run_full_gurobi.ps1
```

## Secondary validation workflow

### 1. Smoke training
```powershell
.\scripts\train_smoke.ps1
```

### 2. CBC smoke tests
```powershell
.\scripts\run_smoke_cbc.ps1
```

## Direct Python entry points

### Training
```powershell
python -m levis_original_mip.training --model mnist_mlp --out_dir checkpoints --data_dir data --device cpu --epochs 3 --batch_size 128 --lr 1e-3 --train_subset 12000 --test_subset 2000
```

### Exact MIP
```powershell
python -m levis_original_mip.exact_mip --checkpoint checkpoints\mnist_mlp.pt --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\mnist_mlp_full_gurobi.json
```

## Design choices

The CNN models are intentionally MIP-friendly:
- no max-pooling
- no batch normalization
- only `Conv2d`, `ReLU`, `Flatten`, and `Linear`

This keeps the exact affine/ReLU structure straightforward to encode.

## Output files

### Training outputs
- `checkpoints\*.pt`
- `checkpoints\*_summary.json`

### Exact MIP outputs
- `results\*.json`
- `smoke_results\*.json`

## Notes

- The implementation currently solves **one exact instance per invocation** of `exact_mip.py`.
- CBC is mainly for smoke tests.
- Gurobi is the intended backend for serious runs.
- The project is packaged with a standard `src/` layout so another user can install it with `pip install -e .`.

## GitHub publishing

After files are written locally, you can publish with:

```powershell
.\scripts\publish_to_github.ps1
```
