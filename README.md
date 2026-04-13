# Original LEVIS MIP: Exact ReLU Verification for MNIST and CIFAR-10

This repository is a structured Python project for training small ReLU networks and solving the **original exact targeted \(\ell_\infty\)** mixed-integer formulation for nearest-adversarial-point search.

The project supports four model families:

- `mnist_mlp`
- `mnist_cnn`
- `cifar10_mlp`
- `cifar10_cnn`

The exact MIP supports two solver backends:

- **CBC** for smoke tests and portability
- **Gurobi** for serious full runs

## Project layout

```text
src/levis_original_mip/
  __init__.py
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
  publish_to_github.ps1
```

## What the code does

### Training
`training.py` trains one of the supported MNIST or CIFAR-10 models and saves a checkpoint with:

- `model_name`
- `state_dict`
- `best_test_acc`
- training history

### Exact MIP
`exact_mip.py` builds the original exact targeted \(\ell_\infty\) formulation:

- decision variable: perturbed input `x`
- objective: minimize `t`
- constraints: `|x - x0| <= t`
- exact affine layer equations for `Linear` and `Conv2d`
- exact Big-M ReLU constraints with one binary per unstable ReLU
- targeted margin-violation constraint:
  `logit[true_class] - logit[target_class] <= 0`

### Smoke runner
`run_smoke_tests.py` looks for one correctly classified test example per model and runs a single exact MIP instance.

## Recommended usage

### 1. Create the virtual environment and install the project
```powershell
.\scripts\setup_env.ps1
```

### 2. Run the small smoke-training pass
```powershell
.\scripts\train_smoke.ps1
```

### 3. Run CBC smoke tests
```powershell
.\scripts\run_smoke_cbc.ps1
```

### 4. Run full training for handoff / real compute
```powershell
.\scripts\train_full.ps1
```

### 5. Run full exact Gurobi jobs
```powershell
.\scripts\run_full_gurobi.ps1
```

## Notes

- The CNNs are intentionally MIP-friendly:
  - no max-pooling
  - no batch normalization
  - only `Conv2d`, `ReLU`, `Flatten`, and `Linear`
- The current implementation solves **one exact instance per invocation** of `exact_mip.py`.
- Gurobi is the intended backend for full experiments.
- CBC is included for sanity checks and easy installation.

## Example direct Python entry points

```powershell
python -m levis_original_mip.training --model mnist_mlp --out_dir checkpoints --data_dir data --device cpu --epochs 3 --batch_size 128 --lr 1e-3 --train_subset 12000 --test_subset 2000
python -m levis_original_mip.exact_mip --checkpoint checkpoints\mnist_mlp.pt --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\mnist_mlp_full_gurobi.json
```

## Status

The package structure is meant for handoff and GitHub publishing. The smoke path is useful for validation, but the **main intended path** is:

1. full training
2. exact Gurobi runs
3. collect JSON results in `results/`

