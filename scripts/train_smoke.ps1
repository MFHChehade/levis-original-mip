$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot
& .\.venv\Scripts\Activate.ps1

New-Item -ItemType Directory -Force -Path checkpoints | Out-Null

python -m levis_original_mip.training --model mnist_mlp --out_dir checkpoints --data_dir data --device cpu --epochs 3 --batch_size 128 --lr 1e-3 --train_subset 12000 --test_subset 2000
python -m levis_original_mip.training --model mnist_cnn --out_dir checkpoints --data_dir data --device cpu --epochs 5 --batch_size 128 --lr 1e-3 --train_subset 12000 --test_subset 2000
python -m levis_original_mip.training --model cifar10_mlp --out_dir checkpoints --data_dir data --device cpu --epochs 6 --batch_size 128 --lr 1e-3 --train_subset 20000 --test_subset 4000
python -m levis_original_mip.training --model cifar10_cnn --out_dir checkpoints --data_dir data --device cpu --epochs 8 --batch_size 128 --lr 1e-3 --train_subset 20000 --test_subset 4000
