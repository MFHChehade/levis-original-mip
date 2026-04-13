$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Split-Path -Parent $Root
Set-Location $Repo
& .\.venv\Scripts\Activate.ps1

New-Item -ItemType Directory -Force -Path results | Out-Null

python -m levis_original_mip.exact_mip --checkpoint checkpoints\mnist_mlp.pt   --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\mnist_mlp_full_gurobi.json
python -m levis_original_mip.exact_mip --checkpoint checkpoints\mnist_cnn.pt   --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\mnist_cnn_full_gurobi.json
python -m levis_original_mip.exact_mip --checkpoint checkpoints\cifar10_mlp.pt --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\cifar10_mlp_full_gurobi.json
python -m levis_original_mip.exact_mip --checkpoint checkpoints\cifar10_cnn.pt --data_dir data --time_limit 300 --mip_gap 0.01 --max_candidates 1000 --solver gurobi --out_json results\cifar10_cnn_full_gurobi.json

