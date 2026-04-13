# Project Status

## Current state

This repository is a **handoff-ready exact-MIP baseline implementation** for small MNIST and CIFAR-10 ReLU networks.

The code already supports:
- training four small reference models
- saving checkpoints
- exact targeted L-infinity MIP construction
- a dual-backend interface: `cbc` and `gurobi`
- scripted exact runs through `exact_mip.py`

## What has already been validated

Successful CBC smoke tests were completed for:
- `mnist_mlp`
- `mnist_cnn`
- `cifar10_mlp`
- `cifar10_cnn`

That confirms:
- checkpoint loading works
- sample selection works
- MIP construction works
- the backend switch works
- JSON result export works

## Intended actual-use path

For real experiments, use:
- **full-dataset training**
- **Gurobi** exact runs
- larger time limits
- scripted repetition over more samples

## Limitations

This repository currently solves one exact targeted instance per invocation. It is not yet a multi-sample benchmark harness. That is acceptable for a clean handoff baseline because scripted outer loops can handle repeated calls.

