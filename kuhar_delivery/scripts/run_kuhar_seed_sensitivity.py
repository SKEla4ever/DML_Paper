#!/usr/bin/env python3
"""Run and summarize KU-HAR selected-method seed sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("outputs/kuhar_seed_sensitivity_v1")
DEFAULT_SEEDS = [20260615, 20260616, 20260617]


@dataclass(frozen=True)
class MethodConfig:
    name: str
    script: Path
    fixed_args: tuple[str, ...]
    existing_outputs: dict[int, Path]


METHODS = [
    MethodConfig(
        name="FedAvg",
        script=Path("algorithms/1d_cnn_fedavg/train_kuhar_1d_cnn_fedavg.py"),
        fixed_args=(
            "--cohort",
            "minimum_support",
            "--rounds",
            "50",
            "--eval-every",
            "10",
            "--client-fraction",
            "1.0",
            "--local-epochs",
            "2",
            "--batch-size",
            "64",
            "--optimizer",
            "adam",
            "--lr",
            "0.001",
            "--norm",
            "batchnorm",
        ),
        existing_outputs={
            20260615: Path(
                "outputs/kuhar_1d_cnn_fedavg_tuning_v1/"
                "batchnorm_adam_lr0p001_e2_r50"
            )
        },
    ),
    MethodConfig(
        name="FedProx",
        script=Path("algorithms/fedprox/train_kuhar_fedprox.py"),
        fixed_args=(
            "--cohort",
            "minimum_support",
            "--rounds",
            "50",
            "--eval-every",
            "10",
            "--client-fraction",
            "1.0",
            "--local-epochs",
            "2",
            "--batch-size",
            "64",
            "--optimizer",
            "adam",
            "--lr",
            "0.001",
            "--norm",
            "batchnorm",
            "--mu",
            "0.1",
        ),
        existing_outputs={20260615: Path("outputs/kuhar_fedprox_mu0p1_r50")},
    ),
    MethodConfig(
        name="SCAFFOLD",
        script=Path("algorithms/scaffold/train_kuhar_scaffold.py"),
        fixed_args=(
            "--cohort",
            "minimum_support",
            "--rounds",
            "50",
            "--eval-every",
            "10",
            "--client-fraction",
            "1.0",
            "--local-epochs",
            "2",
            "--batch-size",
            "64",
            "--lr",
            "1.0",
            "--norm",
            "batchnorm",
        ),
        existing_outputs={20260615: Path("outputs/kuhar_scaffold_lr1p0_r50")},
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_dir(output_root: Path, method: MethodConfig, seed: int) -> Path:
    method_slug = method.name.lower()
    return output_root / f"{method_slug}_seed{seed}"


def selected_run_dir(output_root: Path, method: MethodConfig, seed: int) -> Path:
    existing = method.existing_outputs.get(seed)
    if existing is not None and (REPO_ROOT / existing / "final_metrics.json").exists():
        return existing
    return seed_dir(output_root, method, seed)


def run_missing(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        for seed in args.seeds:
            run_dir = selected_run_dir(args.output_root, method, seed)
            final_metrics = REPO_ROOT / run_dir / "final_metrics.json"
            if final_metrics.exists() and not args.force:
                print(f"skip existing {method.name} seed={seed}: {run_dir}")
                continue
            command = [
                sys.executable,
                str(REPO_ROOT / method.script),
                *method.fixed_args,
                "--seed",
                str(seed),
                "--output-dir",
                str(REPO_ROOT / seed_dir(args.output_root, method, seed)),
            ]
            print(f"run {method.name} seed={seed}")
            print(" ".join(map(str, command)))
            if not args.dry_run:
                subprocess.run(command, check=True, cwd=REPO_ROOT)


def load_result(method: MethodConfig, seed: int, run_dir: Path) -> dict[str, object]:
    config = json.loads((REPO_ROOT / run_dir / "run_config.json").read_text())
    final = json.loads((REPO_ROOT / run_dir / "final_metrics.json").read_text())
    metrics = final["metrics"]
    communication = final["communication"]
    validation = metrics["validation"]
    test = metrics["test"]
    resolved_run_dir = (REPO_ROOT / run_dir).resolve()
    try:
        display_run_dir = str(resolved_run_dir.relative_to(REPO_ROOT))
    except ValueError:
        display_run_dir = str(resolved_run_dir)
    return {
        "method": method.name,
        "seed": seed,
        "run_dir": display_run_dir,
        "rounds": config["rounds"],
        "local_epochs": config["local_epochs"],
        "batch_size": config["batch_size"],
        "optimizer": config.get("optimizer", "corrected_sgd"),
        "lr": config["lr"],
        "mu": config.get("mu", ""),
        "norm": config["norm"],
        "total_communication_bytes": communication["total_bytes"],
        "validation_accuracy": validation["accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "validation_user_mean_macro_f1": validation["per_user"]["mean_macro_f1"],
        "validation_user_worst10_macro_f1": validation["per_user"][
            "worst_10pct_macro_f1"
        ],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_user_mean_macro_f1": test["per_user"]["mean_macro_f1"],
        "test_user_worst10_macro_f1": test["per_user"]["worst_10pct_macro_f1"],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_names = [
        "validation_macro_f1",
        "validation_user_mean_macro_f1",
        "validation_user_worst10_macro_f1",
        "test_macro_f1",
        "test_user_mean_macro_f1",
        "test_user_worst10_macro_f1",
        "total_communication_bytes",
    ]
    summary_rows = []
    for method in [method.name for method in METHODS]:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        summary = {
            "method": method,
            "n_seeds": len(method_rows),
            "seeds": ",".join(str(row["seed"]) for row in method_rows),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in method_rows]
            mean, std = mean_std(values)
            summary[f"{metric_name}_mean"] = mean
            summary[f"{metric_name}_std"] = std
        summary_rows.append(summary)
    summary_rows.sort(key=lambda row: row["validation_macro_f1_mean"], reverse=True)
    return summary_rows


def collect_results(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    for method in METHODS:
        for seed in args.seeds:
            run_dir = selected_run_dir(args.output_root, method, seed)
            final_metrics = REPO_ROOT / run_dir / "final_metrics.json"
            if not final_metrics.exists():
                raise FileNotFoundError(f"missing final metrics: {final_metrics}")
            rows.append(load_result(method, seed, run_dir))
    rows.sort(key=lambda row: (row["method"], row["seed"]))
    summary_rows = summarize(rows)

    write_csv(args.output_root / "seed_results.csv", rows)
    write_csv(args.output_root / "seed_summary.csv", summary_rows)
    (args.output_root / "seed_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    (args.output_root / "seed_summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True) + "\n"
    )
    return rows, summary_rows


def main() -> None:
    args = parse_args()
    args.output_root = REPO_ROOT / args.output_root
    run_missing(args)
    if args.dry_run:
        return
    rows, summary_rows = collect_results(args)
    print(json.dumps({"results": rows, "summary": summary_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
