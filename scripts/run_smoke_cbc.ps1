$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
.\.venv\Scripts\Activate.ps1

New-Item -ItemType Directory -Force -Path smoke_results | Out-Null

python .\run_smoke_tests.py --root_dir . --data_dir data --time_limit 15 --mip_gap 0.05 --max_candidates 1000 --solver cbc

