# Project status

This repository is structured as a proper Python project for handoff and external execution.

Main path:
1. create environment
2. install package
3. train full models
4. run exact Gurobi jobs

Secondary path:
1. train smoke models
2. run CBC smoke tests

The current implementation solves one exact targeted \(\ell_\infty\) instance per invocation of `exact_mip.py`.
