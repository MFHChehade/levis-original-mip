
import argparse
import json
from pathlib import Path

from .exact_mip import solve_single_instance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--time_limit", type=float, default=15.0)
    parser.add_argument("--mip_gap", type=float, default=0.05)
    parser.add_argument("--max_candidates", type=int, default=1000)
    parser.add_argument("--solver", choices=["cbc", "gurobi"], default="cbc")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    ckpt_dir = root_dir / "checkpoints"
    out_dir = root_dir / "smoke_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["mnist_mlp", "mnist_cnn", "cifar10_mlp", "cifar10_cnn"]
    all_results = []

    for model_name in model_names:
        ckpt = ckpt_dir / f"{model_name}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        result = solve_single_instance(
            checkpoint_path=str(ckpt),
            data_dir=args.data_dir,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            max_candidates=args.max_candidates,
            solver=args.solver,
        )
        out_path = out_dir / f"{model_name}_smoke_{args.solver}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        all_results.append(result)

    summary_path = out_dir / f"all_smoke_results_{args.solver}.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
