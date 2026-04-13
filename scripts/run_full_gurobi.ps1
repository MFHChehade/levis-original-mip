$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
.\.venv\Scripts\Activate.ps1

New-Item -ItemType Directory -Force -Path results | Out-Null

python .\exact_mip.py --checkpoint checkpoints\mnist_mlp.pt   --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\mnist_mlp_gurobi.json
python .\exact_mip.py --checkpoint checkpoints\mnist_cnn.pt   --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\mnist_cnn_gurobi.json
python .\exact_mip.py --checkpoint checkpoints\cifar10_mlp.pt --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\cifar10_mlp_gurobi.json
python .\exact_mip.py --checkpoint checkpoints\cifar10_cnn.pt --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\cifar10_cnn_gurobi.json

