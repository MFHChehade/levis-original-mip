# Project status

This repository is organized as a proper Python package for handoff.

## Intended main path
- full training via `scripts/train_full.ps1`
- full exact runs via `scripts/run_full_gurobi.ps1`

## Secondary path
- smoke training via `scripts/train_smoke.ps1`
- CBC smoke MIPs via `scripts/run_smoke_cbc.ps1`

## Solver policy
- CBC: smoke / sanity checking
- Gurobi: full exact runs

