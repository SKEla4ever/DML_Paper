#!/usr/bin/env python3
"""Run the final validation-only HHAR FedAvg LR-schedule comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = REPO_ROOT / "hhar_delivery/scripts/run_hhar_fedavg_3split_tuning_v1.py"
SPEC = importlib.util.spec_from_file_location("hhar_fedavg_tuning_v1", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load tuning base at {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)


DEFAULT_SOURCE_SUMMARY = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_50round_refinement_v1_summary.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/hhar_fedavg_lr_schedule_v1")
DEFAULT_PREREGISTRATION = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_lr_schedule_v1_preregistration.json"
)
DEFAULT_REPORT = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_lr_schedule_v1.md"
)
DEFAULT_PUBLISHED_SUMMARY = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_lr_schedule_v1_summary.json"
)
TARGET_ROUNDS = 50
LOCAL_EPOCHS = 1
SCHEDULE_CANDIDATES = (
    {
        "config_id": "batchnorm_sgd_lr0p01_constant",
        "norm": "batchnorm",
        "optimizer": "sgd",
        "lr": 0.01,
        "momentum": 0.9,
        "schedule_label": "constant 0.01",
    },
    {
        "config_id": "batchnorm_sgd_lr0p03_step20_gamma0p1",
        "norm": "batchnorm",
        "optimizer": "sgd",
        "lr": 0.03,
        "momentum": 0.9,
        "lr_schedule": "step",
        "lr_step_rounds": [20],
        "lr_step_gamma": 0.1,
        "lr_min": 0.0,
        "schedule_label": "0.03, then 0.003 after round 20",
    },
    {
        "config_id": "batchnorm_sgd_lr0p03_cosine_min0p001",
        "norm": "batchnorm",
        "optimizer": "sgd",
        "lr": 0.03,
        "momentum": 0.9,
        "lr_schedule": "cosine",
        "lr_step_rounds": [],
        "lr_step_gamma": 0.1,
        "lr_min": 0.001,
        "schedule_label": "cosine 0.03 to 0.001 over 50 rounds",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=base.DEFAULT_ARCHIVE)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--training-python", type=Path, default=base.DEFAULT_TRAINING_PYTHON
    )
    parser.add_argument("--split-root", type=Path, default=base.DEFAULT_SPLIT_ROOT)
    parser.add_argument(
        "--split-summary", type=Path, default=base.DEFAULT_SPLIT_SUMMARY
    )
    parser.add_argument(
        "--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY
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
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError(f"could not identify source commit for {path}")
    return commit


def source_facts(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    summary = base.read_json(args.source_summary)
    selected = summary["selected_high_budget"]
    efficient = summary["communication_efficient_reference"]
    if selected["config_id"] != "batchnorm_sgd_lr0p01":
        raise RuntimeError("unexpected source high-budget configuration")
    trigger = {
        "communication_efficient_config": efficient["config_id"],
        "communication_efficient_rounds": efficient["rounds"],
        "communication_efficient_validation_mean_macro_f1": efficient[
            "validation_mean_macro_f1"
        ],
        "constant_lr_high_budget_config": selected["config_id"],
        "constant_lr_high_budget_rounds": selected["rounds"],
        "constant_lr_high_budget_validation_mean_macro_f1": selected[
            "validation_mean_macro_f1"
        ],
        "high_budget_minus_efficient_validation_mean_macro_f1": float(
            selected["validation_mean_macro_f1"]
        )
        - float(efficient["validation_mean_macro_f1"]),
    }
    return summary, trigger


def preregistration(
    args: argparse.Namespace,
    trigger: dict[str, object],
    runtime: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment": "hhar_fedavg_lr_schedule_v1",
        "source_refinement_summary": str(args.source_summary),
        "source_refinement_summary_sha256": base.file_sha256(args.source_summary),
        "source_refinement_result_commit": source_commit(args.source_summary),
        "schedule_check_trigger": trigger,
        "split_seeds": list(base.SPLIT_SEEDS),
        "fixed_model_seed": base.MODEL_SEED,
        "training_runtime": runtime,
        "fixed_protocol": {
            "rounds": TARGET_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": args.batch_size,
            "client_fraction": args.client_fraction,
            "eval_every": args.eval_every,
            "normalization": "batchnorm",
            "optimizer": "sgd with momentum 0.9",
            "weight_decay": 0.0,
        },
        "candidates": list(SCHEDULE_CANDIDATES),
        "selection_rule": (
            "At exactly 50 rounds, find the maximum 3-split mean validation "
            f"Macro-F1. Treat candidates within {base.PRACTICAL_TIE} as a practical "
            "tie, then choose lower sample SD, higher worst-split Macro-F1, and "
            "finally earlier frozen candidate order."
        ),
        "selection_boundary": {
            "evaluation_splits": list(base.EVALUATION_SPLITS),
            "test_metrics_generated": False,
            "pair_level_metrics_used_for_selection": False,
        },
        "stop_rule": (
            "This is the final FedAvg optimization search for HHAR V1. Freeze the "
            "selected 50-round candidate without adding further LR schedules."
        ),
    }


def existing_constant_rows(
    args: argparse.Namespace,
    config: dict[str, object],
    split_identity: dict[int, dict[str, object]],
) -> tuple[list[dict[str, object]], dict[int, Path]]:
    source_config = dict(config)
    source_config["config_id"] = "batchnorm_sgd_lr0p01"
    source_config.pop("schedule_label")
    rows = []
    directories = {}
    for split_seed in base.SPLIT_SEEDS:
        output_dir = (
            Path("outputs/hhar_fedavg_50round_refinement_v1")
            / "candidate"
            / "batchnorm_sgd_lr0p01_e1_r50"
            / f"split_seed{split_seed}"
        )
        row = base.validate_run(
            args,
            output_dir,
            source_config,
            LOCAL_EPOCHS,
            TARGET_ROUNDS,
            split_seed,
            split_identity,
        )
        row["config_id"] = config["config_id"]
        rows.append(row)
        directories[split_seed] = output_dir
    return rows, directories


def run_all(
    args: argparse.Namespace,
    split_identity: dict[int, dict[str, object]],
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, dict[int, Path]],
]:
    rows_by_config = {}
    dirs_by_config = {}
    new_index = 0
    for candidate_index, config in enumerate(SCHEDULE_CANDIDATES):
        config_id = str(config["config_id"])
        if candidate_index == 0:
            rows, directories = existing_constant_rows(
                args, config, split_identity
            )
            print("validated existing constant-lr reference", flush=True)
        else:
            rows = []
            directories = {}
            for split_seed in base.SPLIT_SEEDS:
                new_index += 1
                row = base.run_one(
                    args,
                    "schedule",
                    config,
                    LOCAL_EPOCHS,
                    TARGET_ROUNDS,
                    split_seed,
                    split_identity,
                    f"schedule {new_index}/6",
                )
                rows.append(row)
                directories[split_seed] = Path(str(row["run_dir"]))
        rows_by_config[config_id] = rows
        dirs_by_config[config_id] = directories
    return rows_by_config, dirs_by_config


def trajectory_with_lr(
    directories: dict[int, Path], config: dict[str, object]
) -> list[dict[str, object]]:
    by_round: dict[int, list[tuple[float, int, float]]] = {}
    for output_dir in directories.values():
        for record in base.read_json(output_dir / "metrics_history.json"):
            round_index = int(record["round"])
            by_round.setdefault(round_index, []).append(
                (
                    float(record["metrics"]["validation"]["macro_f1"]),
                    int(record["communication"]["total_bytes"]),
                    float(record.get("learning_rate", config["lr"])),
                )
            )
    rows = []
    for round_index in sorted(by_round):
        values = [value for value, _communication, _lr in by_round[round_index]]
        communications = {
            communication
            for _value, communication, _lr in by_round[round_index]
        }
        learning_rates = {lr for _value, _communication, lr in by_round[round_index]}
        if len(values) != 3 or len(communications) != 1 or len(learning_rates) != 1:
            raise RuntimeError(f"inconsistent trajectory at round {round_index}")
        rows.append(
            {
                "round": round_index,
                "effective_learning_rate": next(iter(learning_rates)),
                "validation_mean_macro_f1": statistics.mean(values),
                "validation_sample_std_macro_f1": base.sample_std(values),
                "validation_min_macro_f1": min(values),
                "validation_max_macro_f1": max(values),
                "total_communication_bytes": next(iter(communications)),
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
    source_summary: dict[str, object],
    summaries: list[dict[str, object]],
    selected: dict[str, object],
    tie_set: list[str],
    trajectory: list[dict[str, object]],
    devices: list[dict[str, object]],
) -> None:
    labels = {
        str(config["config_id"]): str(config["schedule_label"])
        for config in SCHEDULE_CANDIDATES
    }
    candidate_table = [
        [
            "yes" if row["config_id"] == selected["config_id"] else "",
            labels[str(row["config_id"])],
            f"{fmt(row['validation_mean_macro_f1'])} +/- "
            f"{fmt(row['validation_sample_std_macro_f1'])}",
            fmt(row["validation_min_macro_f1"]),
            fmt(row["validation_range_macro_f1"]),
            f"{int(row['total_communication_bytes']):,}",
        ]
        for row in sorted(
            summaries,
            key=lambda item: float(item["validation_mean_macro_f1"]),
            reverse=True,
        )
    ]
    trajectory_table = [
        [
            row["round"],
            f"{float(row['effective_learning_rate']):.6f}",
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
    efficient = source_summary["communication_efficient_reference"]
    lines = [
        "# HHAR FedAvg Validation-Only LR-Schedule Selection V1",
        "",
        "## Purpose",
        "",
        "This final bounded search checks whether a standard learning-rate decay "
        "produces a stronger 50-round FedAvg baseline than the selected constant-LR "
        "reference. It uses the same three frozen splits and no test metrics.",
        "",
        "## Frozen Candidate Set",
        "",
        "- Constant `0.01` reference from the target-budget refinement.",
        "- Step decay: `0.03` through round 20, then `0.003`.",
        "- Cosine decay: `0.03` to `0.001` over 50 rounds.",
        "- BatchNorm, SGD momentum `0.9`, 1 local epoch, full participation.",
        "- Stop rule: no additional HHAR FedAvg optimization candidates after this "
        "comparison.",
        "",
        "## Results",
        "",
        markdown_table(
            [
                "Selected",
                "LR schedule",
                "Val mean +/- SD",
                "Worst split",
                "Range",
                "Communication",
            ],
            candidate_table,
        ),
        "",
        f"Practical-tie set: `{tie_set}`. Frozen 50-round FedAvg config: "
        f"`{selected['config_id']}`.",
        "",
        "## Selected Trajectory",
        "",
        markdown_table(
            [
                "Round",
                "Effective LR",
                "Val mean +/- SD",
                "Worst split",
                "Communication",
            ],
            trajectory_table,
        ),
        "",
        "## Validation By Device",
        "",
        "These metrics are descriptive and were not used for selection.",
        "",
        markdown_table(
            ["Device", "Val mean +/- SD", "Range"], device_table
        ),
        "",
        "## Frozen Baseline",
        "",
        "Primary high-budget HHAR FedAvg reference:",
        "",
        f"- Config: `{selected['config_id']}`.",
        "- Rounds: `50`; local epochs: `1`; full participation.",
        f"- Validation Macro-F1: "
        f"`{fmt(selected['validation_mean_macro_f1'])} +/- "
        f"{fmt(selected['validation_sample_std_macro_f1'])}`.",
        f"- Communication: `{int(selected['total_communication_bytes']):,}` bytes.",
        "",
        "Communication-efficient supplementary reference:",
        "",
        f"- Config: `{efficient['config_id']}` at 20 rounds.",
        f"- Validation Macro-F1: "
        f"`{fmt(efficient['validation_mean_macro_f1'])} +/- "
        f"{fmt(efficient['validation_sample_std_macro_f1'])}`.",
        f"- Communication: `{int(efficient['total_communication_bytes']):,}` bytes.",
        "",
        "The two budgets remain separate. Subsequent methods must use the matching "
        "round and communication budget. All tuning histories contain only `train` "
        "and `validation` performance metrics.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    source_summary, trigger = source_facts(args)
    split_identity = base.load_split_identity(args)
    args.training_runtime = base.training_runtime(args)
    registered = preregistration(args, trigger, args.training_runtime)
    if args.preregistration_path.exists():
        existing = base.read_json(args.preregistration_path)
        if existing != registered:
            raise RuntimeError(
                "tracked pre-registration differs; create a new protocol version"
            )
    base.write_json(args.preregistration_path, registered)
    base.write_json(args.output_root / "pre_registration.json", registered)
    if args.dry_run:
        print(json.dumps(registered, indent=2, sort_keys=True))
        return

    rows_by_config, dirs_by_config = run_all(args, split_identity)
    summaries = [
        base.aggregate_runs(rows_by_config[str(config["config_id"])], config)
        for config in SCHEDULE_CANDIDATES
    ]
    order = {
        str(config["config_id"]): index
        for index, config in enumerate(SCHEDULE_CANDIDATES)
    }
    selected, tie_set = base.select_pilot(summaries, order)
    selected_id = str(selected["config_id"])
    selected_config = next(
        config
        for config in SCHEDULE_CANDIDATES
        if config["config_id"] == selected_id
    )
    trajectory = trajectory_with_lr(
        dirs_by_config[selected_id], selected_config
    )
    devices = base.validation_device_summary(dirs_by_config[selected_id])
    all_rows = [
        row
        for config in SCHEDULE_CANDIDATES
        for row in rows_by_config[str(config["config_id"])]
    ]
    if any(bool(row["test_metrics_generated"]) for row in all_rows):
        raise RuntimeError("test metric leakage detected")

    base.write_csv(args.output_root / "candidate_results.csv", all_rows)
    base.write_csv(args.output_root / "candidate_summary.csv", summaries)
    base.write_csv(args.output_root / "selected_trajectory.csv", trajectory)
    base.write_csv(args.output_root / "validation_device_summary.csv", devices)
    summary = {
        "pre_registration": registered,
        "test_access_audit": {
            "evaluation_splits": list(base.EVALUATION_SPLITS),
            "test_metrics_generated": False,
            "all_candidate_runs_validated": True,
        },
        "candidate_results": all_rows,
        "candidate_summary": summaries,
        "practical_tie_set": tie_set,
        "selected_high_budget": selected,
        "selected_trajectory": trajectory,
        "validation_device_summary": devices,
        "communication_efficient_reference": source_summary[
            "communication_efficient_reference"
        ],
    }
    base.write_json(args.output_root / "summary.json", summary)
    base.write_json(args.published_summary_path, summary)
    write_report(
        args.report_path,
        registered,
        source_summary,
        summaries,
        selected,
        tie_set,
        trajectory,
        devices,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
