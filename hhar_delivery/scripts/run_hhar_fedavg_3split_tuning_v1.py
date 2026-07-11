#!/usr/bin/env python3
"""Run pre-registered, validation-only HHAR FedAvg tuning over three splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = Path("algorithms/1d_cnn_fedavg/train_hhar_1d_cnn_fedavg.py")
DEFAULT_ARCHIVE = Path("hhar_delivery/data/raw/hhar/Activity recognition exp.zip")
DEFAULT_CACHE_DIR = Path("outputs/cache")
DEFAULT_TRAINING_PYTHON = Path(".venv/bin/python")
DEFAULT_SPLIT_ROOT = Path("outputs/hhar_split_seed_sensitivity_v1")
DEFAULT_SPLIT_SUMMARY = Path(
    "hhar_delivery/reports/hhar/hhar_split_seed_sensitivity_v1_summary.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/hhar_fedavg_3split_tuning_v1")
DEFAULT_PREREGISTRATION = Path(
    "hhar_delivery/reports/hhar/"
    "hhar_fedavg_3split_tuning_v1_preregistration.json"
)
DEFAULT_REPORT = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_3split_tuning_v1.md"
)
DEFAULT_PUBLISHED_SUMMARY = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_3split_tuning_v1_summary.json"
)
SPLIT_SEEDS = (20260615, 20260616, 20260617)
MODEL_SEED = 20260615
EVALUATION_SPLITS = ("train", "validation")
PRACTICAL_TIE = 0.005
LOCAL_EPOCH_WORST_SPLIT_TOLERANCE = 0.01
PILOT_GRID = (
    {
        "config_id": "batchnorm_adam_lr0p001",
        "norm": "batchnorm",
        "optimizer": "adam",
        "lr": 0.001,
        "momentum": 0.0,
    },
    {
        "config_id": "batchnorm_adam_lr0p0003",
        "norm": "batchnorm",
        "optimizer": "adam",
        "lr": 0.0003,
        "momentum": 0.0,
    },
    {
        "config_id": "groupnorm_adam_lr0p001",
        "norm": "groupnorm",
        "optimizer": "adam",
        "lr": 0.001,
        "momentum": 0.0,
    },
    {
        "config_id": "groupnorm_adam_lr0p0003",
        "norm": "groupnorm",
        "optimizer": "adam",
        "lr": 0.0003,
        "momentum": 0.0,
    },
    {
        "config_id": "batchnorm_sgd_lr0p03",
        "norm": "batchnorm",
        "optimizer": "sgd",
        "lr": 0.03,
        "momentum": 0.9,
    },
    {
        "config_id": "batchnorm_sgd_lr0p01",
        "norm": "batchnorm",
        "optimizer": "sgd",
        "lr": 0.01,
        "momentum": 0.9,
    },
    {
        "config_id": "groupnorm_sgd_lr0p03",
        "norm": "groupnorm",
        "optimizer": "sgd",
        "lr": 0.03,
        "momentum": 0.9,
    },
    {
        "config_id": "groupnorm_sgd_lr0p01",
        "norm": "groupnorm",
        "optimizer": "sgd",
        "lr": 0.01,
        "momentum": 0.9,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--training-python", type=Path, default=DEFAULT_TRAINING_PYTHON
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--split-summary", type=Path, default=DEFAULT_SPLIT_SUMMARY
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--preregistration-path", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--published-summary-path",
        type=Path,
        default=DEFAULT_PUBLISHED_SUMMARY,
    )
    parser.add_argument("--pilot-rounds", type=int, default=20)
    parser.add_argument("--endpoint-rounds", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def training_runtime(args: argparse.Namespace) -> dict[str, object]:
    command = [
        str(args.training_python),
        "-c",
        (
            "import json,platform,numpy,torch; "
            "print(json.dumps({'python_version': platform.python_version(), "
            "'numpy_version': numpy.__version__, "
            "'torch_version': torch.__version__, "
            "'mps_available': bool(torch.backends.mps.is_available())}, "
            "sort_keys=True))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"training Python is unavailable or lacks dependencies: "
            f"{args.training_python}\n{completed.stderr.strip()}"
        )
    runtime = json.loads(completed.stdout)
    runtime["training_python"] = str(args.training_python)
    return runtime


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_split_identity(args: argparse.Namespace) -> dict[int, dict[str, object]]:
    summary = read_json(args.split_summary)
    metadata = {
        int(row["split_seed"]): dict(row) for row in summary["run_metadata"]
    }
    if tuple(sorted(metadata)) != SPLIT_SEEDS:
        raise RuntimeError(
            f"split sensitivity identity is {sorted(metadata)}, expected {SPLIT_SEEDS}"
        )
    split_rows = {
        int(row["split_seed"]): row
        for row in summary["split_results"]
        if row["protocol"] == "centralized_oracle"
    }
    for split_seed, row in metadata.items():
        row["split_windows"] = {
            "train": int(split_rows[split_seed]["train_windows"]),
            "validation": int(split_rows[split_seed]["validation_windows"]),
            "test": int(split_rows[split_seed]["test_windows"]),
        }
        if int(row["model_seed"]) != MODEL_SEED:
            raise RuntimeError("split sensitivity used an unexpected model seed")
    return metadata


def preregistration(
    args: argparse.Namespace,
    split_identity: dict[int, dict[str, object]],
    runtime: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment": "hhar_fedavg_3split_tuning_v1",
        "source_split_summary": str(args.split_summary),
        "source_split_summary_sha256": file_sha256(args.split_summary),
        "split_seeds": list(SPLIT_SEEDS),
        "split_identity": [split_identity[seed] for seed in SPLIT_SEEDS],
        "fixed_model_seed": MODEL_SEED,
        "training_runtime": runtime,
        "selection_boundary": {
            "evaluation_splits": list(EVALUATION_SPLITS),
            "test_metrics_generated_during_tuning": False,
            "selection_metric": "mean validation global Macro-F1 over 3 splits",
            "pair_level_metrics_used_for_selection": False,
        },
        "fixed_training_protocol": {
            "model": "1D CNN",
            "modality": "phone accelerometer",
            "clients": "physical-user/device pairs",
            "client_fraction": args.client_fraction,
            "batch_size": args.batch_size,
            "weight_decay": 0.0,
            "groupnorm_groups": 8,
            "eval_every": args.eval_every,
            "pilot_rounds": args.pilot_rounds,
            "endpoint_rounds": args.endpoint_rounds,
        },
        "stage_1_pilot": {
            "local_epochs": 1,
            "grid": list(PILOT_GRID),
            "selection_rule": (
                "Find the maximum 3-split mean validation Macro-F1. Treat configs "
                f"within {PRACTICAL_TIE} as a practical tie, then choose lower "
                "sample SD, higher worst-split Macro-F1, and finally earlier frozen "
                "grid order."
            ),
        },
        "stage_2_local_epochs": {
            "candidates": [1, 2],
            "applied_to": "stage_1 selected norm/optimizer/lr",
            "selection_rule": (
                "Use 2 local epochs only if its 3-split mean validation Macro-F1 "
                f"improves by more than {PRACTICAL_TIE} and its worst split is not "
                f"lower by more than {LOCAL_EPOCH_WORST_SPLIT_TOLERANCE}; otherwise "
                "retain 1 local epoch."
            ),
        },
        "stage_3_endpoint": {
            "rounds": args.endpoint_rounds,
            "selection": "fixed endpoint, not selected from test or trajectory",
            "reproduction_check": (
                "The independently rerun round-20 validation checkpoints must match "
                "the selected stage-1/stage-2 checkpoints within 1e-6."
            ),
        },
    }


def manifest_dir(args: argparse.Namespace, split_seed: int) -> Path:
    return args.split_root / f"split_seed{split_seed}" / "manifest"


def run_dir(
    args: argparse.Namespace,
    stage: str,
    config_id: str,
    local_epochs: int,
    rounds: int,
    split_seed: int,
) -> Path:
    return (
        args.output_root
        / stage
        / f"{config_id}_e{local_epochs}_r{rounds}"
        / f"split_seed{split_seed}"
    )


def expected_run_config(
    args: argparse.Namespace,
    config: dict[str, object],
    local_epochs: int,
    rounds: int,
    split_seed: int,
) -> dict[str, object]:
    expected = {
        "dataset": "HHAR",
        "manifest_version": "V1",
        "manifest_dir": str(manifest_dir(args, split_seed)),
        "archive": str(args.archive),
        "cache_dir": str(args.cache_dir),
        "rounds": rounds,
        "client_fraction": args.client_fraction,
        "local_epochs": local_epochs,
        "batch_size": args.batch_size,
        "lr": config["lr"],
        "momentum": config["momentum"],
        "weight_decay": 0.0,
        "optimizer": config["optimizer"],
        "norm": config["norm"],
        "groupnorm_groups": 8,
        "eval_every": args.eval_every,
        "evaluation_splits": list(EVALUATION_SPLITS),
        "seed": MODEL_SEED,
        "python_version": args.training_runtime["python_version"],
        "torch_version": args.training_runtime["torch_version"],
    }
    for key in ("lr_schedule", "lr_step_rounds", "lr_step_gamma", "lr_min"):
        if key in config:
            expected[key] = config[key]
    return expected


def validate_run(
    args: argparse.Namespace,
    output_dir: Path,
    config: dict[str, object],
    local_epochs: int,
    rounds: int,
    split_seed: int,
    split_identity: dict[int, dict[str, object]],
) -> dict[str, object]:
    run_config = read_json(output_dir / "run_config.json")
    expected = expected_run_config(
        args, config, local_epochs, rounds, split_seed
    )
    mismatches = [
        f"run_config.{key}: expected {value!r}, got {run_config.get(key)!r}"
        for key, value in expected.items()
        if run_config.get(key) != value
    ]
    if args.device != "auto" and run_config.get("resolved_device") != args.device:
        mismatches.append(
            "resolved device does not match the explicitly requested device"
        )
    manifest_sanity = read_json(output_dir / "manifest_sanity.json")
    if manifest_sanity["split_windows"] != split_identity[split_seed]["split_windows"]:
        mismatches.append("manifest split counts do not match frozen split identity")

    final = read_json(output_dir / "final_metrics.json")
    history = read_json(output_dir / "metrics_history.json")
    if int(final["round"]) != rounds:
        mismatches.append(f"final round is {final['round']!r}, expected {rounds}")
    expected_metric_splits = set(EVALUATION_SPLITS)
    if set(final["metrics"]) != expected_metric_splits:
        mismatches.append(
            f"final metric splits are {sorted(final['metrics'])}, "
            f"expected {sorted(expected_metric_splits)}"
        )
    if any(set(record["metrics"]) != expected_metric_splits for record in history):
        mismatches.append("history contains a non-registered evaluation split")
    round_rows = read_csv(output_dir / "round_metrics.csv")
    if any(key.startswith("test_") for key in round_rows[0]):
        mismatches.append("round_metrics.csv contains test metrics")
    group_rows = read_csv(output_dir / "group_metrics.csv")
    if {row["split"] for row in group_rows} != expected_metric_splits:
        mismatches.append("group_metrics.csv contains an unexpected split")
    if mismatches:
        raise RuntimeError(
            f"invalid tuning output at {output_dir}:\n- " + "\n- ".join(mismatches)
        )
    validation = final["metrics"]["validation"]
    return {
        "stage": output_dir.parts[-3],
        "config_id": config["config_id"],
        "norm": config["norm"],
        "optimizer": config["optimizer"],
        "lr": config["lr"],
        "momentum": config["momentum"],
        "local_epochs": local_epochs,
        "rounds": rounds,
        "split_seed": split_seed,
        "model_seed": MODEL_SEED,
        "resolved_device": run_config["resolved_device"],
        "validation_accuracy": validation["accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "validation_pair_mean_macro_f1_descriptive": validation["per_user"][
            "mean_macro_f1"
        ],
        "total_communication_bytes": final["communication"]["total_bytes"],
        "test_metrics_generated": False,
        "run_dir": str(output_dir),
    }


def run_one(
    args: argparse.Namespace,
    stage: str,
    config: dict[str, object],
    local_epochs: int,
    rounds: int,
    split_seed: int,
    split_identity: dict[int, dict[str, object]],
    ordinal: str,
) -> dict[str, object]:
    output_dir = run_dir(
        args,
        stage,
        str(config["config_id"]),
        local_epochs,
        rounds,
        split_seed,
    )
    final_path = output_dir / "final_metrics.json"
    if final_path.exists() and not args.force:
        row = validate_run(
            args,
            output_dir,
            config,
            local_epochs,
            rounds,
            split_seed,
            split_identity,
        )
        print(f"[{ordinal}] validated existing {output_dir}", flush=True)
        return row
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.training_python),
        str(TRAINER),
        "--manifest-dir",
        str(manifest_dir(args, split_seed)),
        "--archive",
        str(args.archive),
        "--cache-dir",
        str(args.cache_dir),
        "--output-dir",
        str(output_dir),
        "--rounds",
        str(rounds),
        "--eval-every",
        str(args.eval_every),
        "--client-fraction",
        str(args.client_fraction),
        "--local-epochs",
        str(local_epochs),
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
        "--evaluation-splits",
        *EVALUATION_SPLITS,
        "--device",
        args.device,
        "--seed",
        str(MODEL_SEED),
    ]
    if "lr_schedule" in config:
        command.extend(["--lr-schedule", str(config["lr_schedule"])])
        command.extend(
            ["--lr-step-gamma", str(config.get("lr_step_gamma", 0.1))]
        )
        command.extend(["--lr-min", str(config.get("lr_min", 0.0))])
        step_rounds = [str(value) for value in config.get("lr_step_rounds", [])]
        if step_rounds:
            command.extend(["--lr-step-rounds", *step_rounds])
    print(f"[{ordinal}] running {output_dir}", flush=True)
    with (output_dir / "run.log").open("w") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = (output_dir / "run.log").read_text(errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"training failed with code {completed.returncode}: {' '.join(command)}\n"
            + "\n".join(tail)
        )
    return validate_run(
        args,
        output_dir,
        config,
        local_epochs,
        rounds,
        split_seed,
        split_identity,
    )


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_runs(
    rows: list[dict[str, object]], config: dict[str, object]
) -> dict[str, object]:
    values = [float(row["validation_macro_f1"]) for row in rows]
    communications = {int(row["total_communication_bytes"]) for row in rows}
    devices = {str(row["resolved_device"]) for row in rows}
    if len(rows) != 3 or len(communications) != 1 or len(devices) != 1:
        raise RuntimeError(
            f"incomplete or inconsistent 3-split aggregate for {config['config_id']}"
        )
    first = rows[0]
    return {
        "config_id": config["config_id"],
        "norm": config["norm"],
        "optimizer": config["optimizer"],
        "lr": config["lr"],
        "momentum": config["momentum"],
        "local_epochs": first["local_epochs"],
        "rounds": first["rounds"],
        "n_split_seeds": len(values),
        "validation_mean_macro_f1": statistics.mean(values),
        "validation_sample_std_macro_f1": sample_std(values),
        "validation_min_macro_f1": min(values),
        "validation_max_macro_f1": max(values),
        "validation_range_macro_f1": max(values) - min(values),
        "total_communication_bytes": next(iter(communications)),
        "resolved_device": next(iter(devices)),
    }


def select_pilot(
    summaries: list[dict[str, object]], grid_order: dict[str, int]
) -> tuple[dict[str, object], list[str]]:
    best_mean = max(float(row["validation_mean_macro_f1"]) for row in summaries)
    eligible = [
        row
        for row in summaries
        if best_mean - float(row["validation_mean_macro_f1"]) <= PRACTICAL_TIE
    ]
    eligible.sort(
        key=lambda row: (
            float(row["validation_sample_std_macro_f1"]),
            -float(row["validation_min_macro_f1"]),
            grid_order[str(row["config_id"])],
        )
    )
    return eligible[0], [str(row["config_id"]) for row in eligible]


def endpoint_trajectory(
    endpoint_dirs: dict[int, Path], endpoint_rounds: int
) -> list[dict[str, object]]:
    by_round: dict[int, list[tuple[float, int]]] = {}
    for split_seed, output_dir in endpoint_dirs.items():
        history = read_json(output_dir / "metrics_history.json")
        for record in history:
            round_index = int(record["round"])
            by_round.setdefault(round_index, []).append(
                (
                    float(record["metrics"]["validation"]["macro_f1"]),
                    int(record["communication"]["total_bytes"]),
                )
            )
    if endpoint_rounds not in by_round:
        raise RuntimeError("endpoint history is incomplete")
    rows = []
    for round_index in sorted(by_round):
        values = [value for value, _communication in by_round[round_index]]
        communications = {
            communication for _value, communication in by_round[round_index]
        }
        if len(values) != 3 or len(communications) != 1:
            raise RuntimeError(f"inconsistent trajectory at round {round_index}")
        rows.append(
            {
                "round": round_index,
                "validation_mean_macro_f1": statistics.mean(values),
                "validation_sample_std_macro_f1": sample_std(values),
                "validation_min_macro_f1": min(values),
                "validation_max_macro_f1": max(values),
                "total_communication_bytes": next(iter(communications)),
            }
        )
    return rows


def validation_device_summary(
    endpoint_dirs: dict[int, Path]
) -> list[dict[str, object]]:
    by_device: dict[str, list[float]] = {}
    for output_dir in endpoint_dirs.values():
        for row in read_csv(output_dir / "group_metrics.csv"):
            if row["split"] == "validation" and row["grouping"] == "device":
                by_device.setdefault(row["group_id"], []).append(
                    float(row["macro_f1"])
                )
    rows = []
    for device_id, values in sorted(by_device.items()):
        if len(values) != 3:
            raise RuntimeError(f"device {device_id} is missing a split result")
        rows.append(
            {
                "device_id": device_id,
                "validation_mean_macro_f1": statistics.mean(values),
                "validation_sample_std_macro_f1": sample_std(values),
                "validation_min_macro_f1": min(values),
                "validation_max_macro_f1": max(values),
                "validation_range_macro_f1": max(values) - min(values),
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def fmt(value: object) -> str:
    return f"{float(value):.4f}"


def write_report(
    path: Path,
    prereg: dict[str, object],
    pilot_summaries: list[dict[str, object]],
    pilot_selected: dict[str, object],
    pilot_tied: list[str],
    local_summaries: list[dict[str, object]],
    selected_local_epochs: int,
    local_decision: dict[str, object],
    endpoint_rows: list[dict[str, object]],
    endpoint_summary: dict[str, object],
    trajectory: list[dict[str, object]],
    devices: list[dict[str, object]],
) -> None:
    pilot_table = [
        [
            "yes" if row["config_id"] == pilot_selected["config_id"] else "",
            row["config_id"],
            row["norm"],
            row["optimizer"],
            row["lr"],
            f"{fmt(row['validation_mean_macro_f1'])} +/- "
            f"{fmt(row['validation_sample_std_macro_f1'])}",
            fmt(row["validation_min_macro_f1"]),
            fmt(row["validation_range_macro_f1"]),
            f"{int(row['total_communication_bytes']):,}",
        ]
        for row in sorted(
            pilot_summaries,
            key=lambda item: float(item["validation_mean_macro_f1"]),
            reverse=True,
        )
    ]
    local_table = [
        [
            row["local_epochs"],
            f"{fmt(row['validation_mean_macro_f1'])} +/- "
            f"{fmt(row['validation_sample_std_macro_f1'])}",
            fmt(row["validation_min_macro_f1"]),
            fmt(row["validation_range_macro_f1"]),
            f"{int(row['total_communication_bytes']):,}",
            "yes" if int(row["local_epochs"]) == selected_local_epochs else "",
        ]
        for row in local_summaries
    ]
    endpoint_table = [
        [
            row["split_seed"],
            fmt(row["validation_macro_f1"]),
            f"{int(row['total_communication_bytes']):,}",
        ]
        for row in endpoint_rows
    ]
    trajectory_table = [
        [
            row["round"],
            f"{fmt(row['validation_mean_macro_f1'])} +/- "
            f"{fmt(row['validation_sample_std_macro_f1'])}",
            fmt(row["validation_min_macro_f1"]),
            f"{int(row['total_communication_bytes']):,}",
        ]
        for row in trajectory
    ]
    device_table = [
        [
            row["device_id"],
            f"{fmt(row['validation_mean_macro_f1'])} +/- "
            f"{fmt(row['validation_sample_std_macro_f1'])}",
            fmt(row["validation_range_macro_f1"]),
        ]
        for row in devices
    ]
    round_20 = next(row for row in trajectory if int(row["round"]) == 20)
    endpoint_gain = float(endpoint_summary["validation_mean_macro_f1"]) - float(
        round_20["validation_mean_macro_f1"]
    )
    lines = [
        "# HHAR FedAvg Three-Split Validation-Only Tuning V1",
        "",
        "## Selection Boundary",
        "",
        "This staged pass selects a standard global FedAvg baseline using only "
        "validation metrics aggregated over the three frozen execution splits. "
        "No test prediction, test performance metric, or test-driven checkpoint "
        "choice is produced during tuning.",
        "",
        "## Pre-Registered Protocol",
        "",
        f"- Split seeds: `{prereg['split_seeds']}`.",
        f"- Fixed model/optimizer seed: `{prereg['fixed_model_seed']}`.",
        "- Pilot: 8 frozen normalization/optimizer/learning-rate configs, 20 "
        "rounds, 1 local epoch, full participation.",
        f"- Practical-tie tolerance: `{PRACTICAL_TIE}` validation Macro-F1.",
        "- Local computation: compare 1 versus 2 local epochs only for the pilot "
        "winner.",
        "- Final endpoint: fixed 50 rounds; trajectory is descriptive and does not "
        "select a checkpoint.",
        "- Global validation Macro-F1 is the selection metric. Pair-level values "
        "are descriptive because 69 user-device pairs come from nine users.",
        "",
        "## Pilot Grid",
        "",
        markdown_table(
            [
                "Selected",
                "Config",
                "Norm",
                "Optimizer",
                "LR",
                "Val mean +/- SD",
                "Worst split",
                "Range",
                "Communication",
            ],
            pilot_table,
        ),
        "",
        f"Practical-tie set: `{pilot_tied}`. Selected pilot config: "
        f"`{pilot_selected['config_id']}`.",
        "",
        "## Local-Epoch Sensitivity",
        "",
        markdown_table(
            [
                "Local epochs",
                "Val mean +/- SD",
                "Worst split",
                "Range",
                "Communication",
                "Selected",
            ],
            local_table,
        ),
        "",
        f"Mean improvement from 1 to 2 local epochs: "
        f"`{fmt(local_decision['mean_improvement'])}`; worst-split change: "
        f"`{fmt(local_decision['worst_split_change'])}`. Selected local epochs: "
        f"`{selected_local_epochs}`.",
        "",
        "## Frozen 50-Round Endpoint",
        "",
        f"- Configuration: `{endpoint_summary['config_id']}`.",
        f"- Local epochs: `{selected_local_epochs}`.",
        f"- Aggregate validation Macro-F1: "
        f"`{fmt(endpoint_summary['validation_mean_macro_f1'])} +/- "
        f"{fmt(endpoint_summary['validation_sample_std_macro_f1'])}`.",
        f"- Validation range: `{fmt(endpoint_summary['validation_range_macro_f1'])}`.",
        f"- Communication: "
        f"`{int(endpoint_summary['total_communication_bytes']):,}` bytes.",
        "",
        markdown_table(
            ["Split seed", "Validation Macro-F1", "Communication"],
            endpoint_table,
        ),
        "",
        "## Validation Trajectory",
        "",
        markdown_table(
            ["Round", "Val mean +/- SD", "Worst split", "Communication"],
            trajectory_table,
        ),
        "",
        "## Validation By Device",
        "",
        "These values are descriptive and were not used for selection.",
        "",
        markdown_table(
            ["Device", "Val mean +/- SD", "Range"], device_table
        ),
        "",
        "## Interpretation",
        "",
        f"The fixed 50-round endpoint changes mean validation Macro-F1 by "
        f"`{fmt(endpoint_gain)}` relative to its independently reproduced "
        "round-20 checkpoint. The endpoint remains fixed regardless of whether an "
        "earlier trajectory point is numerically higher.",
        "",
        "Every machine-readable training history contains exactly `train` and "
        "`validation` metrics. The tuning code rejects any run containing a test "
        "metric, so the selected setting remains test-blind.",
        "",
        "This is split sensitivity over only nine physical users, not an estimate "
        "based on 69 independent users. Model-seed sensitivity must be run "
        "separately after freezing this configuration.",
        "",
        "## Artifacts",
        "",
        "- Local run results: "
        "`outputs/hhar_fedavg_3split_tuning_v1/run_results.csv`",
        "- Local pilot summary: "
        "`outputs/hhar_fedavg_3split_tuning_v1/pilot_summary.csv`",
        "- Published summary: "
        "`hhar_delivery/reports/hhar/"
        "hhar_fedavg_3split_tuning_v1_summary.json`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.pilot_rounds != 20 or args.endpoint_rounds != 50:
        raise ValueError("V1 requires exactly 20 pilot and 50 endpoint rounds")
    split_identity = load_split_identity(args)
    args.training_runtime = training_runtime(args)
    registered = preregistration(args, split_identity, args.training_runtime)
    if args.preregistration_path.exists():
        existing = read_json(args.preregistration_path)
        if existing != registered:
            raise RuntimeError(
                "tracked pre-registration differs; create a new protocol version"
            )
    write_json(args.preregistration_path, registered)
    write_json(args.output_root / "pre_registration.json", registered)
    if args.dry_run:
        print(json.dumps(registered, indent=2, sort_keys=True))
        return

    all_run_rows: list[dict[str, object]] = []
    pilot_by_config: dict[str, list[dict[str, object]]] = {}
    total_pilot_runs = len(PILOT_GRID) * len(SPLIT_SEEDS)
    pilot_index = 0
    for config in PILOT_GRID:
        config_rows = []
        for split_seed in SPLIT_SEEDS:
            pilot_index += 1
            row = run_one(
                args,
                "pilot",
                config,
                1,
                args.pilot_rounds,
                split_seed,
                split_identity,
                f"pilot {pilot_index}/{total_pilot_runs}",
            )
            config_rows.append(row)
            all_run_rows.append(row)
        pilot_by_config[str(config["config_id"])] = config_rows

    grid_order = {
        str(config["config_id"]): index
        for index, config in enumerate(PILOT_GRID)
    }
    pilot_summaries = [
        aggregate_runs(pilot_by_config[str(config["config_id"])], config)
        for config in PILOT_GRID
    ]
    pilot_selected, pilot_tied = select_pilot(pilot_summaries, grid_order)
    selected_config = next(
        config
        for config in PILOT_GRID
        if config["config_id"] == pilot_selected["config_id"]
    )
    print(f"pilot selected {selected_config['config_id']}", flush=True)

    local_epoch_1 = pilot_by_config[str(selected_config["config_id"])]
    local_epoch_2 = []
    for index, split_seed in enumerate(SPLIT_SEEDS, start=1):
        row = run_one(
            args,
            "local_epoch_sensitivity",
            selected_config,
            2,
            args.pilot_rounds,
            split_seed,
            split_identity,
            f"local epochs {index}/3",
        )
        local_epoch_2.append(row)
        all_run_rows.append(row)
    local_summaries = [
        aggregate_runs(local_epoch_1, selected_config),
        aggregate_runs(local_epoch_2, selected_config),
    ]
    e1, e2 = local_summaries
    mean_improvement = float(e2["validation_mean_macro_f1"]) - float(
        e1["validation_mean_macro_f1"]
    )
    worst_split_change = float(e2["validation_min_macro_f1"]) - float(
        e1["validation_min_macro_f1"]
    )
    selected_local_epochs = (
        2
        if mean_improvement > PRACTICAL_TIE
        and worst_split_change >= -LOCAL_EPOCH_WORST_SPLIT_TOLERANCE
        else 1
    )
    local_decision = {
        "mean_improvement": mean_improvement,
        "worst_split_change": worst_split_change,
        "selected_local_epochs": selected_local_epochs,
    }
    print(f"selected local_epochs={selected_local_epochs}", flush=True)

    endpoint_rows = []
    endpoint_dirs: dict[int, Path] = {}
    for index, split_seed in enumerate(SPLIT_SEEDS, start=1):
        row = run_one(
            args,
            "endpoint",
            selected_config,
            selected_local_epochs,
            args.endpoint_rounds,
            split_seed,
            split_identity,
            f"endpoint {index}/3",
        )
        endpoint_rows.append(row)
        all_run_rows.append(row)
        endpoint_dirs[split_seed] = Path(str(row["run_dir"]))
    endpoint_summary = aggregate_runs(endpoint_rows, selected_config)
    trajectory = endpoint_trajectory(endpoint_dirs, args.endpoint_rounds)

    selected_stage_rows = (
        local_epoch_2 if selected_local_epochs == 2 else local_epoch_1
    )
    selected_stage_by_seed = {
        int(row["split_seed"]): float(row["validation_macro_f1"])
        for row in selected_stage_rows
    }
    checkpoint_20_differences = {}
    for split_seed, output_dir in endpoint_dirs.items():
        history = read_json(output_dir / "metrics_history.json")
        checkpoint = next(record for record in history if int(record["round"]) == 20)
        difference = abs(
            float(checkpoint["metrics"]["validation"]["macro_f1"])
            - selected_stage_by_seed[split_seed]
        )
        checkpoint_20_differences[str(split_seed)] = difference
        if difference > 1e-6:
            raise RuntimeError(
                f"round-20 reproduction failed for split seed {split_seed}: "
                f"difference={difference}"
            )

    devices = validation_device_summary(endpoint_dirs)
    resolved_devices = {str(row["resolved_device"]) for row in all_run_rows}
    if len(resolved_devices) != 1:
        raise RuntimeError("resolved device changed during tuning")

    write_csv(args.output_root / "run_results.csv", all_run_rows)
    write_csv(args.output_root / "pilot_summary.csv", pilot_summaries)
    write_csv(args.output_root / "local_epoch_summary.csv", local_summaries)
    write_csv(args.output_root / "endpoint_results.csv", endpoint_rows)
    write_csv(args.output_root / "endpoint_trajectory.csv", trajectory)
    write_csv(args.output_root / "validation_device_summary.csv", devices)
    summary = {
        "pre_registration": registered,
        "test_access_audit": {
            "evaluation_splits": list(EVALUATION_SPLITS),
            "test_metrics_generated": False,
            "all_completed_runs_validated": True,
        },
        "resolved_device": next(iter(resolved_devices)),
        "pilot_run_results": [
            row for row in all_run_rows if row["stage"] == "pilot"
        ],
        "pilot_summary": pilot_summaries,
        "pilot_practical_tie_set": pilot_tied,
        "pilot_selected": pilot_selected,
        "local_epoch_run_results": local_epoch_1 + local_epoch_2,
        "local_epoch_summary": local_summaries,
        "local_epoch_decision": local_decision,
        "endpoint_results": endpoint_rows,
        "endpoint_summary": endpoint_summary,
        "endpoint_trajectory": trajectory,
        "validation_device_summary": devices,
        "round_20_reproduction_absolute_differences": checkpoint_20_differences,
    }
    write_json(args.output_root / "summary.json", summary)
    write_json(args.published_summary_path, summary)
    write_report(
        args.report_path,
        registered,
        pilot_summaries,
        pilot_selected,
        pilot_tied,
        local_summaries,
        selected_local_epochs,
        local_decision,
        endpoint_rows,
        endpoint_summary,
        trajectory,
        devices,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
