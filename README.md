# Original LEVIS MIP for MNIST and CIFAR-10

A handoff-ready repository for the **original exact ReLU mixed-integer formulation** used to search for nearest adversarial points on small MNIST and CIFAR-10 networks.

This repository is organized for a collaborator who wants to run the **actual exact MIP pipeline with Gurobi** on stronger hardware. The CBC path is kept only as a secondary install / sanity path and is not the main advertised workflow.

## Scope

This repository implements the **original exact nearest-adversarial-point MIP** for small ReLU networks:

- decision variable is the perturbed input
- objective minimizes the perturbation size using an exact targeted **L-infinity** formulation
- affine layers are modeled exactly
- ReLU nonlinearities are encoded with **exact big-M binary constraints**
- the solver searches for a nearest targeted adversarial point for a correctly classified sample

This repository is **not** the reduced-MIP / complementarity shortcut from LEVIS. It is the direct exact-MIP baseline path.

## Included model families

Four small verification-friendly reference networks are included.

### MNIST
- `mnist_mlp`
  - Flatten -> Linear(784, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 10)
- `mnist_cnn`
  - Conv2d(1, 4, kernel=4, stride=4) -> ReLU
  - Conv2d(4, 4, kernel=3, stride=1, padding=1) -> ReLU
  - Flatten -> Linear(196, 32) -> ReLU -> Linear(32, 10)

### CIFAR-10
- `cifar10_mlp`
  - Flatten -> Linear(3072, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 10)
- `cifar10_cnn`
  - Conv2d(3, 4, kernel=4, stride=4) -> ReLU
  - Conv2d(4, 4, kernel=3, stride=1, padding=1) -> ReLU
  - Flatten -> Linear(256, 32) -> ReLU -> Linear(32, 10)

These architectures intentionally avoid max-pooling, normalization layers, and other extras so the exact MIP mapping stays transparent.

## Main workflow

The intended workflow is:

1. create environment
2. train all four models on the full datasets
3. run exact targeted MIPs with **Gurobi**
4. collect JSON outputs for analysis

The CBC runner is included only for smoke tests and debugging.

## Repository status

This codebase is already suitable to hand off to another researcher or collaborator.

What is already working:
- training code for all four models
- checkpoint saving
- exact targeted L-infinity MIP construction
- dual backend interface with `cbc` and `gurobi`
- successful CBC smoke tests on all four reference models

What it is not:
- not a reduced-MIP LEVIS implementation
- not a large benchmark harness that sweeps many samples in one command
- not a polished reproduction package for every experiment in the LEVIS paper

The main exact entry point solves **one exact targeted instance per invocation**. That is still a clean and useful baseline for scripted batch runs on stronger machines.

## Confirmed smoke-test status

The current code has already completed successful CBC smoke tests on all four reference models.

| Model | Solver | Status | Binary count | Runtime (approx.) |
|---|---|---:|---:|---:|
| `mnist_mlp` | CBC | Optimal | 128 | 20 s |
| `mnist_cnn` | CBC | Optimal | 325 | 19 s |
| `cifar10_mlp` | CBC | Optimal | 192 | 107 s |
| `cifar10_cnn` | CBC | Optimal | 544 | 28 s |

These are only sanity checks. The intended actual path is **Gurobi full runs**.

## Repository layout

```text
.
+-- exact_mip.py
+-- models.py
+-- train_classifier.py
+-- run_smoke_tests.py
+-- requirements.txt
+-- .gitignore
+-- PROJECT_STATUS.md
+-- scripts/
    +-- setup_env.ps1
    +-- train_full.ps1
    +-- run_full_gurobi.ps1
    +-- run_smoke_cbc.ps1
    +-- publish_to_github.ps1
    +-- handoff_commands.txt
```

## Environment setup

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

## Full training

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_full.ps1
```

## Full exact runs with Gurobi

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_full_gurobi.ps1
```

## Optional smoke-only path

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_smoke_cbc.ps1
```

## Direct exact-run commands

```powershell
python .\exact_mip.py --checkpoint checkpoints\mnist_mlp.pt   --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\mnist_mlp_gurobi.json
python .\exact_mip.py --checkpoint checkpoints\mnist_cnn.pt   --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\mnist_cnn_gurobi.json
python .\exact_mip.py --checkpoint checkpoints\cifar10_mlp.pt --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\cifar10_mlp_gurobi.json
python .\exact_mip.py --checkpoint checkpoints\cifar10_cnn.pt --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\cifar10_cnn_gurobi.json
```

Increase the time limit and candidate search range for more serious experiments.

## Suggested GitHub positioning

### Repository name
`levis-original-mip`

### Title
**Original LEVIS MIP for MNIST and CIFAR-10: Exact ReLU Verification with Gurobi**

### Suggested short description
Original LEVIS-style exact ReLU MIP baseline for MNIST and CIFAR-10, with Gurobi full runs and CBC smoke tests.

## What to commit

Commit:
- source code
- PowerShell scripts
- `README.md`
- `.gitignore`
- `requirements.txt`
- `PROJECT_STATUS.md`

Do not commit:
- `.venv/`
- `data/`
- `checkpoints/`
- `results/`
- `smoke_results/`
- solver logs
- temporary artifacts

## Practical notes for the handoff recipient

- A valid **Gurobi installation and license** are required for the intended full runs.
- The training code automatically falls back to CPU if CUDA is unavailable.
- The included CNNs are intentionally small and verification-friendly.
- This repository is best viewed as a clean exact-MIP baseline repo that is easy to extend.

