#!/usr/bin/env python3
"""Run a small validation-driven FedAvg tuning grid."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("outputs/kuhar_1d_cnn_fedavg_tuning_v1")


GRID = [
    {"norm": "batchnorm", "optimizer": "adam", "lr": 0.001, "momentum": 0.0},
    {"norm": "batchnorm", "optimizer": "adam", "lr": 0.0003, "momentum": 0.0},
    {"norm": "groupnorm", "optimizer": "adam", "lr": 0.001, "momentum": 0.0},
    {"norm": "groupnorm", "optimizer": "adam", "lr": 0.0003, "momentum": 0.0},
    {"norm": "batchnorm", "optimizer": "sgd", "lr": 0.03, "momentum": 0.9},
    {"norm": "batchnorm", "optimizer": "sgd", "lr": 0.01, "momentum": 0.9},
    {"norm": "groupnorm", "optimizer": "sgd", "lr": 0.03, "momentum": 0.9},
    {"norm": "groupnorm", "optimizer": "sgd", "lr": 0.01, "momentum": 0.9},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--cohort", choices=("minimum_support", "full_sparse"), default="minimum_support")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def config_name(config: dict[str, object]) -> str:
    lr_text = str(config["lr"]).replace(".", "p")
    return f"{config['norm']}_{config['optimizer']}_lr{lr_text}"


def extract_summary(run_dir: Path) -> dict[str, object]:
    with (run_dir / "run_config.json").open() as handle:
        config = json.load(handle)
    with (run_dir / "final_metrics.json").open() as handle:
        final = json.load(handle)
    metrics = final["metrics"]
    communication = final["communication"]
    return {
        "run_dir": str(run_dir),
        "norm": config["norm"],
        "optimizer": config["optimizer"],
        "lr": config["lr"],
        "momentum": config["momentum"],
        "rounds": config["rounds"],
        "eval_every": config["eval_every"],
        "total_communication_bytes": communication["total_bytes"],
        "validation_accuracy": metrics["validation"]["accuracy"],
        "validation_macro_f1": metrics["validation"]["macro_f1"],
        "validation_user_mean_macro_f1": metrics["validation"]["per_user"]["mean_macro_f1"],
        "validation_user_worst10_macro_f1": metrics["validation"]["per_user"]["worst_10pct_macro_f1"],
        "test_accuracy": metrics["test"]["accuracy"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "test_user_mean_macro_f1": metrics["test"]["per_user"]["mean_macro_f1"],
        "test_user_worst10_macro_f1": metrics["test"]["per_user"]["worst_10pct_macro_f1"],
    }


def write_summary(output_root: Path) -> list[dict[str, object]]:
    rows = []
    for final_metrics in sorted(output_root.glob("*/final_metrics.json")):
        rows.append(extract_summary(final_metrics.parent))
    rows.sort(
        key=lambda row: (
            row["validation_macro_f1"],
            row["validation_user_mean_macro_f1"],
        ),
        reverse=True,
    )
    if rows:
        with (output_root / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (output_root / "summary.json").open("w") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return rows


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("train_kuhar_1d_cnn_fedavg.py")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for index, config in enumerate(GRID, start=1):
        run_dir = args.output_root / config_name(config)
        if (run_dir / "final_metrics.json").exists() and not args.force:
            print(f"[{index}/{len(GRID)}] skip existing {run_dir}")
            continue
        command = [
            sys.executable,
            str(script),
            "--cohort",
            args.cohort,
            "--rounds",
            str(args.rounds),
            "--eval-every",
            str(args.eval_every),
            "--client-fraction",
            str(args.client_fraction),
            "--local-epochs",
            str(args.local_epochs),
            "--batch-size",
            str(args.batch_size),
            "--optimizer",
            str(config["optimizer"]),
            "--lr",
            str(config["lr"]),
            "--momentum",
            str(config["momentum"]),
            "--norm",
            str(config["norm"]),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--output-dir",
            str(run_dir),
        ]
        print(f"[{index}/{len(GRID)}] run {run_dir}")
        subprocess.run(command, check=True)
        rows = write_summary(args.output_root)
        if rows:
            best = rows[0]
            print(
                "current best:",
                best["run_dir"],
                "validation_macro_f1=",
                best["validation_macro_f1"],
            )
    rows = write_summary(args.output_root)
    print(json.dumps(rows[:3], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
