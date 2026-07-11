#!/usr/bin/env python3
"""Refine the HHAR FedAvg configuration at the fixed 50-round budget."""

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
    "hhar_delivery/reports/hhar/hhar_fedavg_3split_tuning_v1_summary.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/hhar_fedavg_50round_refinement_v1")
DEFAULT_PREREGISTRATION = Path(
    "hhar_delivery/reports/hhar/"
    "hhar_fedavg_50round_refinement_v1_preregistration.json"
)
DEFAULT_REPORT = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_50round_refinement_v1.md"
)
DEFAULT_PUBLISHED_SUMMARY = Path(
    "hhar_delivery/reports/hhar/"
    "hhar_fedavg_50round_refinement_v1_summary.json"
)
TARGET_ROUNDS = 50
LOCAL_EPOCHS = 1
TOP_K = 3


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
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    summary = base.read_json(args.source_summary)
    ranked = sorted(
        summary["pilot_summary"],
        key=lambda row: float(row["validation_mean_macro_f1"]),
        reverse=True,
    )
    top_ids = [str(row["config_id"]) for row in ranked[:TOP_K]]
    grid = {
        str(config["config_id"]): dict(config) for config in base.PILOT_GRID
    }
    candidates = [grid[config_id] for config_id in top_ids]
    selected_20 = summary["pilot_selected"]
    endpoint_50 = summary["endpoint_summary"]
    endpoint_by_seed = {
        int(row["split_seed"]): float(row["validation_macro_f1"])
        for row in summary["endpoint_results"]
    }
    pilot_selected_rows = {
        int(row["split_seed"]): float(row["validation_macro_f1"])
        for row in summary["pilot_run_results"]
        if row["config_id"] == selected_20["config_id"]
    }
    declines = {
        str(seed): endpoint_by_seed[seed] - pilot_selected_rows[seed]
        for seed in base.SPLIT_SEEDS
    }
    trigger = {
        "selected_20_round_config": selected_20["config_id"],
        "round_20_validation_mean_macro_f1": selected_20[
            "validation_mean_macro_f1"
        ],
        "round_50_validation_mean_macro_f1": endpoint_50[
            "validation_mean_macro_f1"
        ],
        "round_50_minus_round_20_mean_macro_f1": float(
            endpoint_50["validation_mean_macro_f1"]
        )
        - float(selected_20["validation_mean_macro_f1"]),
        "per_split_round_50_minus_round_20": declines,
        "all_three_splits_declined": all(value < 0 for value in declines.values()),
    }
    if not trigger["all_three_splits_declined"]:
        raise RuntimeError("the registered refinement trigger is not satisfied")
    return summary, candidates, trigger


def preregistration(
    args: argparse.Namespace,
    candidates: list[dict[str, object]],
    trigger: dict[str, object],
    runtime: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment": "hhar_fedavg_50round_refinement_v1",
        "source_v1_summary": str(args.source_summary),
        "source_v1_summary_sha256": base.file_sha256(args.source_summary),
        "source_v1_result_commit": source_commit(args.source_summary),
        "refinement_trigger": trigger,
        "split_seeds": list(base.SPLIT_SEEDS),
        "fixed_model_seed": base.MODEL_SEED,
        "training_runtime": runtime,
        "fixed_protocol": {
            "rounds": TARGET_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": args.batch_size,
            "client_fraction": args.client_fraction,
            "eval_every": args.eval_every,
            "normalization": "batchnorm for all top-3 V1 candidates",
            "weight_decay": 0.0,
        },
        "candidate_rule": "top 3 configurations by V1 20-round validation mean",
        "candidates": candidates,
        "selection_rule": (
            "At exactly 50 rounds, find the maximum 3-split mean validation "
            f"Macro-F1. Treat candidates within {base.PRACTICAL_TIE} as a practical "
            "tie, then choose lower sample SD, higher worst-split Macro-F1, and "
            "finally earlier V1 pilot rank."
        ),
        "selection_boundary": {
            "evaluation_splits": list(base.EVALUATION_SPLITS),
            "test_metrics_generated": False,
            "pair_level_metrics_used_for_selection": False,
        },
        "reporting_rule": (
            "Always report the selected 20-round communication-efficient candidate "
            "and selected 50-round high-budget candidate separately. Do not compare "
            "their absolute scores as if communication budgets were equal."
        ),
    }


def existing_high_lr_rows(
    args: argparse.Namespace,
    config: dict[str, object],
    split_identity: dict[int, dict[str, object]],
) -> tuple[list[dict[str, object]], dict[int, Path]]:
    rows = []
    directories = {}
    for split_seed in base.SPLIT_SEEDS:
        output_dir = (
            base.DEFAULT_OUTPUT_ROOT
            / "endpoint"
            / f"{config['config_id']}_e1_r50"
            / f"split_seed{split_seed}"
        )
        rows.append(
            base.validate_run(
                args,
                output_dir,
                config,
                LOCAL_EPOCHS,
                TARGET_ROUNDS,
                split_seed,
                split_identity,
            )
        )
        directories[split_seed] = output_dir
    return rows, directories


def run_candidates(
    args: argparse.Namespace,
    candidates: list[dict[str, object]],
    split_identity: dict[int, dict[str, object]],
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, dict[int, Path]],
]:
    rows_by_config = {}
    dirs_by_config = {}
    total_new = (len(candidates) - 1) * len(base.SPLIT_SEEDS)
    new_index = 0
    for candidate_index, config in enumerate(candidates):
        config_id = str(config["config_id"])
        if candidate_index == 0:
            rows, directories = existing_high_lr_rows(
                args, config, split_identity
            )
            print(f"validated V1 endpoint for {config_id}", flush=True)
        else:
            rows = []
            directories = {}
            for split_seed in base.SPLIT_SEEDS:
                new_index += 1
                row = base.run_one(
                    args,
                    "candidate",
                    config,
                    LOCAL_EPOCHS,
                    TARGET_ROUNDS,
                    split_seed,
                    split_identity,
                    f"refinement {new_index}/{total_new}",
                )
                rows.append(row)
                directories[split_seed] = Path(str(row["run_dir"]))
        rows_by_config[config_id] = rows
        dirs_by_config[config_id] = directories
    return rows_by_config, dirs_by_config


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
    candidate_summaries: list[dict[str, object]],
    selected: dict[str, object],
    tie_set: list[str],
    trajectory: list[dict[str, object]],
    device_summary: list[dict[str, object]],
) -> None:
    candidate_table = [
        [
            "yes" if row["config_id"] == selected["config_id"] else "",
            row["config_id"],
            row["optimizer"],
            row["lr"],
            f"{fmt(row['validation_mean_macro_f1'])} +/- "
            f"{fmt(row['validation_sample_std_macro_f1'])}",
            fmt(row["validation_min_macro_f1"]),
            fmt(row["validation_range_macro_f1"]),
            f"{int(row['total_communication_bytes']):,}",
        ]
        for row in sorted(
            candidate_summaries,
            key=lambda item: float(item["validation_mean_macro_f1"]),
            reverse=True,
        )
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
        for row in device_summary
    ]
    efficient = source_summary["pilot_selected"]
    lines = [
        "# HHAR FedAvg 50-Round Validation-Only Refinement V1",
        "",
        "## Reason",
        "",
        "The pre-registered V1 20-round winner degraded on all three splits when "
        "extended with the same constant learning rate to 50 rounds. This refinement "
        "therefore compares the V1 pilot top three at the target 50-round budget.",
        "",
        f"- V1 round-20 mean: "
        f"`{fmt(prereg['refinement_trigger']['round_20_validation_mean_macro_f1'])}`.",
        f"- V1 round-50 mean: "
        f"`{fmt(prereg['refinement_trigger']['round_50_validation_mean_macro_f1'])}`.",
        f"- Mean change: "
        "`"
        f"{fmt(prereg['refinement_trigger']['round_50_minus_round_20_mean_macro_f1'])}"
        "`.",
        "- Test metrics remain unavailable to selection.",
        "",
        "## Fixed Protocol",
        "",
        "- Candidates: top three V1 pilot configurations.",
        "- Three frozen execution splits; fixed model seed `20260615`.",
        "- 50 rounds, 1 local epoch, batch size 64, full client participation.",
        "- Selection: aggregate validation global Macro-F1 only.",
        f"- Practical-tie tolerance: `{base.PRACTICAL_TIE}`.",
        "",
        "## Target-Budget Results",
        "",
        markdown_table(
            [
                "Selected",
                "Config",
                "Optimizer",
                "LR",
                "Val mean +/- SD",
                "Worst split",
                "Range",
                "Communication",
            ],
            candidate_table,
        ),
        "",
        f"Practical-tie set: `{tie_set}`. Selected high-budget config: "
        f"`{selected['config_id']}`.",
        "",
        "## Selected 50-Round Trajectory",
        "",
        markdown_table(
            ["Round", "Val mean +/- SD", "Worst split", "Communication"],
            trajectory_table,
        ),
        "",
        "## Selected Validation By Device",
        "",
        "These metrics are descriptive and were not used for selection.",
        "",
        markdown_table(
            ["Device", "Val mean +/- SD", "Range"], device_table
        ),
        "",
        "## Frozen FedAvg References",
        "",
        "Communication-efficient reference:",
        "",
        f"- Config: `{efficient['config_id']}`; rounds: `20`; local epochs: `1`.",
        f"- Validation Macro-F1: "
        f"`{fmt(efficient['validation_mean_macro_f1'])} +/- "
        f"{fmt(efficient['validation_sample_std_macro_f1'])}`.",
        f"- Communication: `{int(efficient['total_communication_bytes']):,}` bytes.",
        "",
        "High-budget reference:",
        "",
        f"- Config: `{selected['config_id']}`; rounds: `50`; local epochs: `1`.",
        f"- Validation Macro-F1: "
        f"`{fmt(selected['validation_mean_macro_f1'])} +/- "
        f"{fmt(selected['validation_sample_std_macro_f1'])}`.",
        f"- Communication: `{int(selected['total_communication_bytes']):,}` bytes.",
        "",
        "The two references have different communication budgets and are not used "
        "as a direct absolute-score comparison. Subsequent algorithms must be "
        "compared at the matching 20- or 50-round budget.",
        "",
        "All candidate histories contain only `train` and `validation` performance "
        "metrics. Model-seed sensitivity remains a separate next step.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    source_summary, candidates, trigger = source_facts(args)
    split_identity = base.load_split_identity(args)
    args.training_runtime = base.training_runtime(args)
    registered = preregistration(
        args, candidates, trigger, args.training_runtime
    )
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

    rows_by_config, dirs_by_config = run_candidates(
        args, candidates, split_identity
    )
    candidate_summaries = [
        base.aggregate_runs(rows_by_config[str(config["config_id"])], config)
        for config in candidates
    ]
    order = {
        str(config["config_id"]): index for index, config in enumerate(candidates)
    }
    selected, tie_set = base.select_pilot(candidate_summaries, order)
    selected_id = str(selected["config_id"])
    trajectory = base.endpoint_trajectory(
        dirs_by_config[selected_id], TARGET_ROUNDS
    )
    devices = base.validation_device_summary(dirs_by_config[selected_id])
    all_rows = [
        row
        for config in candidates
        for row in rows_by_config[str(config["config_id"])]
    ]
    if any(bool(row["test_metrics_generated"]) for row in all_rows):
        raise RuntimeError("test metric leakage detected")
    if len({str(row["resolved_device"]) for row in all_rows}) != 1:
        raise RuntimeError("resolved device changed during refinement")

    base.write_csv(args.output_root / "candidate_results.csv", all_rows)
    base.write_csv(
        args.output_root / "candidate_summary.csv", candidate_summaries
    )
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
        "candidate_summary": candidate_summaries,
        "practical_tie_set": tie_set,
        "selected_high_budget": selected,
        "selected_trajectory": trajectory,
        "validation_device_summary": devices,
        "communication_efficient_reference": source_summary["pilot_selected"],
    }
    base.write_json(args.output_root / "summary.json", summary)
    base.write_json(args.published_summary_path, summary)
    write_report(
        args.report_path,
        registered,
        source_summary,
        candidate_summaries,
        selected,
        tie_set,
        trajectory,
        devices,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
