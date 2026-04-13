$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
.\.venv\Scripts\Activate.ps1

New-Item -ItemType Directory -Force -Path checkpoints | Out-Null

python .\train_classifier.py --model mnist_mlp   --out_dir checkpoints --data_dir data --device cuda --epochs 10 --batch_size 256 --lr 1e-3 --train_subset 0 --test_subset 0
python .\train_classifier.py --model mnist_cnn   --out_dir checkpoints --data_dir data --device cuda --epochs 12 --batch_size 256 --lr 1e-3 --train_subset 0 --test_subset 0
python .\train_classifier.py --model cifar10_mlp --out_dir checkpoints --data_dir data --device cuda --epochs 20 --batch_size 256 --lr 1e-3 --train_subset 0 --test_subset 0
python .\train_classifier.py --model cifar10_cnn --out_dir checkpoints --data_dir data --device cuda --epochs 25 --batch_size 256 --lr 1e-3 --train_subset 0 --test_subset 0

