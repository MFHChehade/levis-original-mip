import argparse
import subprocess
import sys
from pathlib import Path

MODELS = ["mnist_mlp", "mnist_cnn", "cifar10_mlp", "cifar10_cnn"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--time_limit", type=int, default=15)
    parser.add_argument("--mip_gap", type=float, default=0.05)
    parser.add_argument("--max_candidates", type=int, default=1000)
    parser.add_argument("--solver", choices=["cbc", "gurobi"], default="cbc")
    args = parser.parse_args()

    root = Path(args.root_dir)
    ckpt_dir = root / "checkpoints"
    out_dir = root / "smoke_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    exact_mip = root / "exact_mip.py"
    if not exact_mip.exists():
        raise FileNotFoundError(f"Missing file: {exact_mip}")

    for model in MODELS:
        ckpt = ckpt_dir / f"{model}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        out_json = out_dir / f"{model}_{args.solver}.json"
        cmd = [
            sys.executable,
            str(exact_mip),
            "--checkpoint", str(ckpt),
            "--data_dir", args.data_dir,
            "--time_limit", str(args.time_limit),
            "--mip_gap", str(args.mip_gap),
            "--max_candidates", str(args.max_candidates),
            "--solver", args.solver,
            "--out_json", str(out_json),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
