#!/usr/bin/env python3
"""Run the one-time evaluation-only locked test for frozen HHAR FedAvg."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPANSION_SCRIPT = (
    REPO_ROOT
    / "hhar_delivery/scripts/run_hhar_fedavg_model_seed_expansion_v1.py"
)
SPEC = importlib.util.spec_from_file_location("hhar_fedavg_seed_expansion_v1", EXPANSION_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load seed expansion runner at {EXPANSION_SCRIPT}")
seed = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = seed
SPEC.loader.exec_module(seed)
base = seed.base


EVALUATOR = Path("hhar_delivery/scripts/evaluate_hhar_fedavg_checkpoint.py")
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "hhar_fedavg_checkpoint_evaluator", REPO_ROOT / EVALUATOR
)
if EVALUATOR_SPEC is None or EVALUATOR_SPEC.loader is None:
    raise RuntimeError(f"could not load checkpoint evaluator at {EVALUATOR}")
checkpoint_evaluator = importlib.util.module_from_spec(EVALUATOR_SPEC)
sys.modules[EVALUATOR_SPEC.name] = checkpoint_evaluator
EVALUATOR_SPEC.loader.exec_module(checkpoint_evaluator)
DEFAULT_SOURCE_SUMMARY = Path(
    "hhar_delivery/reports/hhar/"
    "hhar_fedavg_model_seed_expansion_v1_summary.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/hhar_fedavg_locked_test_v1")
DEFAULT_PREREGISTRATION = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_locked_test_v1_preregistration.json"
)
DEFAULT_REPORT = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_locked_test_v1.md"
)
DEFAULT_PUBLISHED_SUMMARY = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_locked_test_v1_summary.json"
)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260716
MODEL_SEEDS = seed.MODEL_SEEDS
SPLIT_SEEDS = base.SPLIT_SEEDS
DEVELOPMENT_MODEL_SEED = seed.DEVELOPMENT_MODEL_SEED
CONFIRMATORY_MODEL_SEEDS = seed.CONFIRMATORY_MODEL_SEEDS
ACTIVITIES = tuple(checkpoint_evaluator.hhar.ACTIVITIES)


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
    parser.add_argument("--source-summary", type=Path, default=DEFAULT_SOURCE_SUMMARY)
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("mps",), default="mps")
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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


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
) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary = read_json(args.source_summary)
    mismatches = []
    if summary.get("status") != (
        "five_model_seed_characterization_complete_stable_ready_for_locked_test"
    ):
        mismatches.append("source is not frozen as ready for locked test")
    if summary.get("next_step") != (
        "separately pre-register evaluation-only locked test over all 15 "
        "saved checkpoints"
    ):
        mismatches.append("source next-step boundary is unexpected")
    checkpoint_rows = list(summary.get("checkpoint_manifest", []))
    if len(checkpoint_rows) != 15:
        mismatches.append("source does not contain exactly 15 checkpoints")
    source_rows = list(summary.get("run_results", []))
    cells = {(int(row["model_seed"]), int(row["split_seed"])) for row in source_rows}
    expected_cells = {
        (model_seed, split_seed)
        for model_seed in MODEL_SEEDS
        for split_seed in SPLIT_SEEDS
    }
    if cells != expected_cells:
        mismatches.append("source run matrix is not the complete frozen 3x5 design")
    checkpoint_by_cell = {
        (int(row["model_seed"]), int(row["split_seed"])): row
        for row in checkpoint_rows
    }
    if set(checkpoint_by_cell) != expected_cells:
        mismatches.append("checkpoint manifest cells differ from source run cells")
    for source_row in source_rows:
        cell = (int(source_row["model_seed"]), int(source_row["split_seed"]))
        output_dir = Path(str(source_row["run_dir"]))
        checkpoint = output_dir / "final_model.pt"
        final_metrics = output_dir / "final_metrics.json"
        run_config = output_dir / "run_config.json"
        standardization = output_dir / "channel_standardization.json"
        for path in (checkpoint, final_metrics, run_config, standardization):
            if not path.exists():
                mismatches.append(f"missing frozen source artifact: {path}")
        registered = checkpoint_by_cell.get(cell, {})
        if checkpoint.exists():
            if file_sha256(checkpoint) != registered.get("checkpoint_sha256"):
                mismatches.append(f"checkpoint SHA-256 changed for cell {cell}")
            if checkpoint.stat().st_size != registered.get("checkpoint_bytes"):
                mismatches.append(f"checkpoint size changed for cell {cell}")
    if mismatches:
        raise RuntimeError("invalid locked-test source:\n- " + "\n- ".join(mismatches))
    return summary, source_rows


def checkpoint_manifest(
    source_summary: dict[str, object], source_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    source_by_cell = {
        (int(row["model_seed"]), int(row["split_seed"])): row
        for row in source_rows
    }
    output = []
    for registered in source_summary["checkpoint_manifest"]:
        model_seed = int(registered["model_seed"])
        split_seed = int(registered["split_seed"])
        source_row = source_by_cell[(model_seed, split_seed)]
        source_run_dir = Path(str(source_row["run_dir"]))
        checkpoint = source_run_dir / "final_model.pt"
        output.append(
            {
                "model_seed": model_seed,
                "split_seed": split_seed,
                "source_run_dir": str(source_run_dir),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "source_run_config_sha256": file_sha256(
                    source_run_dir / "run_config.json"
                ),
                "source_final_metrics_sha256": file_sha256(
                    source_run_dir / "final_metrics.json"
                ),
                "source_standardization_sha256": file_sha256(
                    source_run_dir / "channel_standardization.json"
                ),
                "validation_macro_f1": source_row["validation_macro_f1"],
            }
        )
    output.sort(key=lambda row: (row["model_seed"], row["split_seed"]))
    return output


def preregistration(
    args: argparse.Namespace,
    source_summary: dict[str, object],
    source_rows: list[dict[str, object]],
    split_identity: dict[int, dict[str, object]],
    runtime: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment": "hhar_fedavg_locked_test_v1",
        "source_result_summary": str(args.source_summary),
        "source_result_summary_sha256": file_sha256(args.source_summary),
        "source_result_commit": source_commit(args.source_summary),
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": file_sha256(EVALUATOR),
        "model_trainer": str(base.TRAINER),
        "model_trainer_sha256": file_sha256(base.TRAINER),
        "training_runtime": runtime,
        "split_identity": [split_identity[split_seed] for split_seed in SPLIT_SEEDS],
        "checkpoint_manifest": checkpoint_manifest(source_summary, source_rows),
        "frozen_configuration": source_summary["frozen_primary_reference"],
        "fixed_evaluation": {
            "evaluation_split": "test",
            "evaluation_only": True,
            "training_performed": False,
            "retraining_prohibited": True,
            "checkpoint_selection_prohibited": True,
            "all_fifteen_checkpoints_required": True,
            "model_seeds": list(MODEL_SEEDS),
            "split_seeds": list(SPLIT_SEEDS),
            "batch_size": args.batch_size,
            "requested_device": args.device,
            "classes": list(ACTIVITIES),
            "expected_test_windows_by_split": {
                str(split_seed): split_identity[split_seed]["split_windows"]["test"]
                for split_seed in SPLIT_SEEDS
            },
        },
        "primary_analysis": {
            "primary_metric": "test global Macro-F1 over all six classes",
            "model_seed_marginal": (
                "For each of five model seeds, average the three split-seed test "
                "scores; report the mean, sample SD, and range across those five "
                "model-seed marginals."
            ),
            "split_seed_marginal": (
                "For each of three split seeds, average five model-seed test scores; "
                "report split marginal sample SD and range separately."
            ),
            "confirmatory_estimate": (
                "Report the mean and sample SD across the four confirmatory "
                "model-seed marginals, excluding development seed 20260615."
            ),
            "factor_decomposition": (
                "Descriptive balanced 3x5 two-way sums-of-squares decomposition "
                "for test Macro-F1; no p-values."
            ),
            "prohibited_summary": (
                "Do not report a pooled 15-cell SD as model-seed or split-seed "
                "uncertainty."
            ),
        },
        "secondary_analysis": {
            "global_metrics": ["loss", "accuracy", "Macro-F1"],
            "validation_to_test_gap": (
                "Test Macro-F1 minus the already frozen validation Macro-F1 for "
                "each matching checkpoint, summarized by model-seed marginals."
            ),
            "per_activity": (
                "Precision, recall, F1, and support for each of six activities; "
                "average splits within each model seed before reporting model-seed "
                "mean, sample SD, and range."
            ),
            "per_device": (
                "Supported-class Macro-F1 per device; average splits within model "
                "seed, then report five-model-seed mean, sample SD, and range."
            ),
            "per_physical_user": (
                "Supported-class Macro-F1 per physical user using physical user as "
                "the independent reporting unit; user-device pairs are descriptive."
            ),
            "confusion": (
                "Average the row-normalized six-class confusion matrices equally "
                "over all 15 cells."
            ),
        },
        "physical_user_bootstrap": {
            "unit": "physical user",
            "score_per_user": (
                "Balanced mean supported-class Macro-F1 over all 15 cells."
            ),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "2.5th and 97.5th percentile of resampled user means",
        },
        "reporting_boundary": {
            "test_used_for_selection": False,
            "test_driven_tuning": False,
            "test_driven_checkpoint_selection": False,
            "all_results_reported": True,
            "physical_user_not_user_device_pair_is_independent_unit": True,
        },
        "stop_rule": (
            "This is the first and final locked-test evaluation for the frozen HHAR "
            "FedAvg baseline. Evaluate every registered checkpoint once, publish all "
            "cells, and perform no post-test retuning or seed selection."
        ),
    }


def cell_output_dir(args: argparse.Namespace, model_seed: int, split_seed: int) -> Path:
    return (
        args.output_root
        / "cells"
        / f"model_seed{model_seed}"
        / f"split_seed{split_seed}"
    )


def source_by_cell(source_rows: list[dict[str, object]]) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (int(row["model_seed"]), int(row["split_seed"])): row
        for row in source_rows
    }


def validate_cell(
    args: argparse.Namespace,
    output_dir: Path,
    registered: dict[str, object],
    source_row: dict[str, object],
    split_identity: dict[int, dict[str, object]],
) -> dict[str, object]:
    model_seed = int(registered["model_seed"])
    split_seed = int(registered["split_seed"])
    evaluation = read_json(output_dir / "evaluation_config.json")
    metrics = read_json(output_dir / "metrics.json")
    mismatches = []
    expected_evaluation = {
        "evaluation_only": True,
        "training_performed": False,
        "split": "test",
        "model_seed": model_seed,
        "split_seed": split_seed,
        "source_run_dir": registered["source_run_dir"],
        "checkpoint": registered["checkpoint"],
        "checkpoint_sha256": registered["checkpoint_sha256"],
        "checkpoint_bytes": registered["checkpoint_bytes"],
        "batch_size": args.batch_size,
        "requested_device": args.device,
        "resolved_device": args.device,
    }
    mismatches.extend(
        f"evaluation_config.{key}: expected {value!r}, got {evaluation.get(key)!r}"
        for key, value in expected_evaluation.items()
        if evaluation.get(key) != value
    )
    authorization = evaluation.get("test_authorization") or {}
    if authorization.get("path") != str(args.preregistration_path):
        mismatches.append("evaluation used a different authorization path")
    if authorization.get("sha256") != file_sha256(args.preregistration_path):
        mismatches.append("evaluation authorization SHA-256 changed")
    global_metrics = metrics["global_metrics"]
    expected_windows = split_identity[split_seed]["split_windows"]["test"]
    if int(global_metrics["windows"]) != expected_windows:
        mismatches.append(
            f"test windows are {global_metrics['windows']}, expected {expected_windows}"
        )
    classes = read_csv(output_dir / "class_metrics.csv")
    if len(classes) != len(ACTIVITIES):
        mismatches.append("class metrics do not contain all six classes")
    if sum(int(row["support"]) for row in classes) != expected_windows:
        mismatches.append("class supports do not sum to test windows")
    class_macro_f1 = statistics.mean(float(row["f1"]) for row in classes)
    if abs(class_macro_f1 - float(global_metrics["macro_f1"])) > 1e-15:
        mismatches.append("class F1 values disagree with global Macro-F1")
    confusion = read_csv(output_dir / "confusion_matrix.csv")
    if len(confusion) != len(ACTIVITIES):
        mismatches.append("confusion matrix does not have six true-class rows")
    if sum(int(row["support"]) for row in confusion) != expected_windows:
        mismatches.append("confusion supports do not sum to test windows")
    groups = read_csv(output_dir / "group_metrics.csv")
    groupings = {row["grouping"] for row in groups}
    if groupings != {"device", "physical_user", "user_device_pair"}:
        mismatches.append(f"unexpected groupings: {sorted(groupings)}")
    if len([row for row in groups if row["grouping"] == "device"]) != 8:
        mismatches.append("test group metrics do not contain all eight devices")
    if len([row for row in groups if row["grouping"] == "physical_user"]) != 9:
        mismatches.append("test group metrics do not contain all nine physical users")
    prediction_path = output_dir / "predictions.csv.gz"
    with gzip.open(prediction_path, "rt", newline="") as handle:
        prediction_count = sum(1 for _row in csv.DictReader(handle))
    if prediction_count != expected_windows:
        mismatches.append(
            f"prediction rows are {prediction_count}, expected {expected_windows}"
        )
    if metrics["checkpoint_identity"]["checkpoint_sha256"] != registered[
        "checkpoint_sha256"
    ]:
        mismatches.append("metrics checkpoint identity differs from registration")
    if mismatches:
        raise RuntimeError(
            f"invalid locked-test cell at {output_dir}:\n- "
            + "\n- ".join(mismatches)
        )
    pair_summary = metrics["user_device_pair_summary_existing_protocol"]
    return {
        "model_seed": model_seed,
        "split_seed": split_seed,
        "source_run_dir": registered["source_run_dir"],
        "checkpoint_sha256": registered["checkpoint_sha256"],
        "test_windows": expected_windows,
        "test_loss": global_metrics["loss"],
        "test_accuracy": global_metrics["accuracy"],
        "test_macro_f1": global_metrics["macro_f1"],
        "validation_macro_f1": source_row["validation_macro_f1"],
        "test_minus_validation_macro_f1": float(global_metrics["macro_f1"])
        - float(source_row["validation_macro_f1"]),
        "test_physical_user_mean_macro_f1": metrics["physical_user_summary"][
            "mean_supported_class_macro_f1"
        ],
        "test_device_mean_macro_f1": metrics["device_summary"][
            "mean_supported_class_macro_f1"
        ],
        "test_user_device_pair_mean_macro_f1_descriptive": pair_summary[
            "mean_macro_f1"
        ],
        "test_user_device_pair_worst10_macro_f1_descriptive": pair_summary[
            "worst_10pct_macro_f1"
        ],
        "resolved_device": evaluation["resolved_device"],
        "training_performed": False,
        "output_dir": str(output_dir),
    }


def run_cell(
    args: argparse.Namespace,
    registered: dict[str, object],
    source_row: dict[str, object],
    split_identity: dict[int, dict[str, object]],
    ordinal: str,
) -> dict[str, object]:
    model_seed = int(registered["model_seed"])
    split_seed = int(registered["split_seed"])
    output_dir = cell_output_dir(args, model_seed, split_seed)
    if (output_dir / "metrics.json").exists() and not args.force:
        row = validate_cell(
            args, output_dir, registered, source_row, split_identity
        )
        print(f"[{ordinal}] validated existing {output_dir}", flush=True)
        return row
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.training_python),
        str(EVALUATOR),
        "--source-run-dir",
        str(registered["source_run_dir"]),
        "--output-dir",
        str(output_dir),
        "--model-seed",
        str(model_seed),
        "--split-seed",
        str(split_seed),
        "--split",
        "test",
        "--archive",
        str(args.archive),
        "--cache-dir",
        str(args.cache_dir),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--test-authorization",
        str(args.preregistration_path),
    ]
    print(f"[{ordinal}] evaluating {output_dir}", flush=True)
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
        tail = (output_dir / "run.log").read_text(errors="replace").splitlines()[-40:]
        raise RuntimeError(
            f"locked-test evaluation failed with code {completed.returncode}: "
            f"{' '.join(command)}\n" + "\n".join(tail)
        )
    return validate_cell(args, output_dir, registered, source_row, split_identity)


def factor_summary(
    rows: list[dict[str, object]], factor: str, values: tuple[int, ...]
) -> list[dict[str, object]]:
    other = "split_seed" if factor == "model_seed" else "model_seed"
    expected_other = SPLIT_SEEDS if factor == "model_seed" else MODEL_SEEDS
    output = []
    for value in values:
        selected = [row for row in rows if int(row[factor]) == value]
        if {int(row[other]) for row in selected} != set(expected_other):
            raise RuntimeError(f"incomplete locked-test matrix for {factor}={value}")
        test_scores = [float(row["test_macro_f1"]) for row in selected]
        validation_scores = [float(row["validation_macro_f1"]) for row in selected]
        gaps = [float(row["test_minus_validation_macro_f1"]) for row in selected]
        output.append(
            {
                factor: value,
                "role": (
                    "development"
                    if factor == "model_seed" and value == DEVELOPMENT_MODEL_SEED
                    else "confirmatory"
                    if factor == "model_seed"
                    else "frozen_split"
                ),
                "n_cells": len(selected),
                "test_mean_macro_f1": statistics.mean(test_scores),
                "test_sample_std_across_other_factor_macro_f1": statistics.stdev(
                    test_scores
                ),
                "test_min_macro_f1": min(test_scores),
                "test_max_macro_f1": max(test_scores),
                "test_range_macro_f1": max(test_scores) - min(test_scores),
                "test_mean_accuracy": statistics.mean(
                    float(row["test_accuracy"]) for row in selected
                ),
                "test_mean_loss": statistics.mean(
                    float(row["test_loss"]) for row in selected
                ),
                "validation_mean_macro_f1": statistics.mean(validation_scores),
                "test_minus_validation_mean_macro_f1": statistics.mean(gaps),
                "test_physical_user_mean_macro_f1": statistics.mean(
                    float(row["test_physical_user_mean_macro_f1"])
                    for row in selected
                ),
                "test_device_mean_macro_f1": statistics.mean(
                    float(row["test_device_mean_macro_f1"]) for row in selected
                ),
                "test_user_device_pair_mean_macro_f1_descriptive": statistics.mean(
                    float(row["test_user_device_pair_mean_macro_f1_descriptive"])
                    for row in selected
                ),
                "test_user_device_pair_worst10_macro_f1_descriptive": statistics.mean(
                    float(row["test_user_device_pair_worst10_macro_f1_descriptive"])
                    for row in selected
                ),
            }
        )
    return output


def marginal_variability(
    summaries: list[dict[str, object]], factor: str
) -> dict[str, object]:
    values = [float(row["test_mean_macro_f1"]) for row in summaries]
    return {
        "factor": factor,
        "n_marginal_means": len(values),
        "balanced_test_mean_macro_f1": statistics.mean(values),
        "marginal_sample_std_macro_f1": statistics.stdev(values),
        "marginal_min_macro_f1": min(values),
        "marginal_max_macro_f1": max(values),
        "marginal_range_macro_f1": max(values) - min(values),
    }


def confirmatory_summary(
    model_summary: list[dict[str, object]],
) -> dict[str, object]:
    by_seed = {int(row["model_seed"]): row for row in model_summary}
    development = float(by_seed[DEVELOPMENT_MODEL_SEED]["test_mean_macro_f1"])
    values = [
        float(by_seed[model_seed]["test_mean_macro_f1"])
        for model_seed in CONFIRMATORY_MODEL_SEEDS
    ]
    return {
        "development_model_seed": DEVELOPMENT_MODEL_SEED,
        "development_test_marginal_mean_macro_f1": development,
        "confirmatory_model_seeds": list(CONFIRMATORY_MODEL_SEEDS),
        "confirmatory_test_mean_of_model_seed_marginals_macro_f1": statistics.mean(
            values
        ),
        "confirmatory_model_seed_marginal_sample_std_macro_f1": statistics.stdev(
            values
        ),
        "confirmatory_min_model_seed_marginal_macro_f1": min(values),
        "confirmatory_max_model_seed_marginal_macro_f1": max(values),
        "confirmatory_minus_development_macro_f1": statistics.mean(values)
        - development,
    }


def two_way_decomposition(rows: list[dict[str, object]]) -> dict[str, object]:
    scores = {
        (int(row["split_seed"]), int(row["model_seed"])): float(
            row["test_macro_f1"]
        )
        for row in rows
    }
    if len(scores) != len(SPLIT_SEEDS) * len(MODEL_SEEDS):
        raise RuntimeError("test decomposition requires the complete 3x5 matrix")
    values = list(scores.values())
    grand = statistics.mean(values)
    split_means = {
        split_seed: statistics.mean(
            [scores[(split_seed, model_seed)] for model_seed in MODEL_SEEDS]
        )
        for split_seed in SPLIT_SEEDS
    }
    model_means = {
        model_seed: statistics.mean(
            [scores[(split_seed, model_seed)] for split_seed in SPLIT_SEEDS]
        )
        for model_seed in MODEL_SEEDS
    }
    ss_total = sum((value - grand) ** 2 for value in values)
    ss_split = len(MODEL_SEEDS) * sum(
        (value - grand) ** 2 for value in split_means.values()
    )
    ss_model = len(SPLIT_SEEDS) * sum(
        (value - grand) ** 2 for value in model_means.values()
    )
    ss_interaction = max(ss_total - ss_split - ss_model, 0.0)
    components = (
        ("split_seed", len(SPLIT_SEEDS) - 1, ss_split),
        ("model_seed", len(MODEL_SEEDS) - 1, ss_model),
        (
            "split_by_model_seed_interaction",
            (len(SPLIT_SEEDS) - 1) * (len(MODEL_SEEDS) - 1),
            ss_interaction,
        ),
    )
    return {
        "method": (
            "balanced 3x5 descriptive decomposition without replicated cells; "
            "interaction is residual and no inferential test is performed"
        ),
        "balanced_test_mean_macro_f1": grand,
        "sources": [
            {
                "source": source,
                "degrees_of_freedom": degrees,
                "sum_of_squares": sum_squares,
                "mean_square": sum_squares / degrees,
                "share_of_total_sum_of_squares": (
                    sum_squares / ss_total if ss_total else 0.0
                ),
            }
            for source, degrees, sum_squares in components
        ],
    }


def collect_class_results(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cell_rows = []
    for row in rows:
        for class_row in read_csv(Path(str(row["output_dir"])) / "class_metrics.csv"):
            cell_rows.append(
                {
                    "model_seed": row["model_seed"],
                    "split_seed": row["split_seed"],
                    "class_id": int(class_row["class_id"]),
                    "class_name": class_row["class_name"],
                    "support": int(class_row["support"]),
                    "precision": float(class_row["precision"]),
                    "recall": float(class_row["recall"]),
                    "f1": float(class_row["f1"]),
                }
            )
    model_rows = []
    final_rows = []
    for class_id, class_name in enumerate(ACTIVITIES):
        for model_seed in MODEL_SEEDS:
            selected = [
                row
                for row in cell_rows
                if row["class_id"] == class_id and row["model_seed"] == model_seed
            ]
            if {row["split_seed"] for row in selected} != set(SPLIT_SEEDS):
                raise RuntimeError(
                    f"incomplete class results for {class_name}, seed {model_seed}"
                )
            model_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "model_seed": model_seed,
                    "mean_precision_across_splits": statistics.mean(
                        row["precision"] for row in selected
                    ),
                    "mean_recall_across_splits": statistics.mean(
                        row["recall"] for row in selected
                    ),
                    "mean_f1_across_splits": statistics.mean(
                        row["f1"] for row in selected
                    ),
                }
            )
        marginals = [
            row
            for row in model_rows
            if row["class_id"] == class_id
        ]
        supports_by_split = {}
        for split_seed in SPLIT_SEEDS:
            supports = {
                row["support"]
                for row in cell_rows
                if row["class_id"] == class_id and row["split_seed"] == split_seed
            }
            if len(supports) != 1:
                raise RuntimeError(
                    f"test support changed across model seeds for {class_name}"
                )
            supports_by_split[split_seed] = next(iter(supports))
        f1_values = [row["mean_f1_across_splits"] for row in marginals]
        final_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "mean_test_windows_per_split": statistics.mean(
                    supports_by_split.values()
                ),
                "min_test_windows_across_splits": min(supports_by_split.values()),
                "max_test_windows_across_splits": max(supports_by_split.values()),
                "mean_precision_over_model_seed_marginals": statistics.mean(
                    row["mean_precision_across_splits"] for row in marginals
                ),
                "mean_recall_over_model_seed_marginals": statistics.mean(
                    row["mean_recall_across_splits"] for row in marginals
                ),
                "mean_f1_over_model_seed_marginals": statistics.mean(f1_values),
                "model_seed_marginal_sample_std_f1": statistics.stdev(f1_values),
                "model_seed_marginal_min_f1": min(f1_values),
                "model_seed_marginal_max_f1": max(f1_values),
                "model_seed_marginal_range_f1": max(f1_values) - min(f1_values),
            }
        )
    return cell_rows, model_rows, final_rows


def collect_group_results(
    rows: list[dict[str, object]], grouping: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cell_rows = []
    for row in rows:
        group_rows = read_csv(Path(str(row["output_dir"])) / "group_metrics.csv")
        for group_row in group_rows:
            if group_row["grouping"] != grouping:
                continue
            cell_rows.append(
                {
                    "model_seed": row["model_seed"],
                    "split_seed": row["split_seed"],
                    "grouping": grouping,
                    "group_id": group_row["group_id"],
                    "windows": int(group_row["windows"]),
                    "supported_classes": int(group_row["supported_classes"]),
                    "accuracy": float(group_row["accuracy"]),
                    "supported_class_macro_f1": float(
                        group_row["supported_class_macro_f1"]
                    ),
                }
            )
    final_rows = []
    for group_id in sorted({row["group_id"] for row in cell_rows}):
        model_marginals = []
        split_counts = []
        class_counts = []
        for model_seed in MODEL_SEEDS:
            selected = [
                row
                for row in cell_rows
                if row["group_id"] == group_id and row["model_seed"] == model_seed
            ]
            if {row["split_seed"] for row in selected} != set(SPLIT_SEEDS):
                raise RuntimeError(
                    f"{grouping} {group_id} is missing from a frozen test split"
                )
            model_marginals.append(
                statistics.mean(row["supported_class_macro_f1"] for row in selected)
            )
            split_counts.extend(row["windows"] for row in selected)
            class_counts.extend(row["supported_classes"] for row in selected)
        final_rows.append(
            {
                "grouping": grouping,
                "group_id": group_id,
                "mean_test_windows_per_cell": statistics.mean(split_counts),
                "min_supported_classes_per_cell": min(class_counts),
                "max_supported_classes_per_cell": max(class_counts),
                "balanced_mean_supported_class_macro_f1": statistics.mean(
                    model_marginals
                ),
                "model_seed_marginal_sample_std_macro_f1": statistics.stdev(
                    model_marginals
                ),
                "model_seed_marginal_min_macro_f1": min(model_marginals),
                "model_seed_marginal_max_macro_f1": max(model_marginals),
                "model_seed_marginal_range_macro_f1": max(model_marginals)
                - min(model_marginals),
            }
        )
    return cell_rows, final_rows


def physical_user_bootstrap(
    physical_user_summary: list[dict[str, object]],
) -> dict[str, object]:
    values = np.asarray(
        [
            float(row["balanced_mean_supported_class_macro_f1"])
            for row in physical_user_summary
        ],
        dtype=np.float64,
    )
    if len(values) != 9:
        raise RuntimeError(f"expected 9 physical users, got {len(values)}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(
        values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True
    ).mean(axis=1)
    return {
        "unit": "physical_user",
        "n_physical_users": len(values),
        "score": "balanced supported-class Macro-F1 over all 15 cells",
        "mean_macro_f1": float(values.mean()),
        "median_macro_f1": float(np.median(values)),
        "population_std_macro_f1": float(values.std()),
        "minimum_macro_f1": float(values.min()),
        "maximum_macro_f1": float(values.max()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_95pct_ci_lower": float(np.percentile(samples, 2.5)),
        "bootstrap_95pct_ci_upper": float(np.percentile(samples, 97.5)),
    }


def mean_normalized_confusion(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    matrices = []
    predicted_columns = [
        f"predicted_{class_id}_{class_name}"
        for class_id, class_name in enumerate(ACTIVITIES)
    ]
    for row in rows:
        confusion_rows = read_csv(
            Path(str(row["output_dir"])) / "confusion_matrix.csv"
        )
        matrix = np.asarray(
            [
                [float(confusion_row[column]) for column in predicted_columns]
                for confusion_row in confusion_rows
            ],
            dtype=np.float64,
        )
        support = matrix.sum(axis=1, keepdims=True)
        if np.any(support == 0):
            raise RuntimeError("a locked test cell has an unsupported global class")
        matrices.append(matrix / support)
    mean_matrix = np.mean(np.stack(matrices), axis=0)
    output = []
    for class_id, class_name in enumerate(ACTIVITIES):
        result: dict[str, object] = {
            "true_class_id": class_id,
            "true_class_name": class_name,
        }
        for predicted_id, predicted_name in enumerate(ACTIVITIES):
            result[f"predicted_{predicted_id}_{predicted_name}"] = float(
                mean_matrix[class_id, predicted_id]
            )
        output.append(result)
    return output


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
    rows: list[dict[str, object]],
    model_summary: list[dict[str, object]],
    split_summary: list[dict[str, object]],
    model_variability: dict[str, object],
    split_variability: dict[str, object],
    confirmation: dict[str, object],
    decomposition: dict[str, object],
    class_summary: list[dict[str, object]],
    device_summary: list[dict[str, object]],
    physical_user_summary: list[dict[str, object]],
    user_bootstrap: dict[str, object],
    confusion: list[dict[str, object]],
) -> None:
    score_by_cell = {
        (int(row["split_seed"]), int(row["model_seed"])): row["test_macro_f1"]
        for row in rows
    }
    matrix_rows = [
        [
            split_seed,
            *[
                fmt(score_by_cell[(split_seed, model_seed)])
                for model_seed in MODEL_SEEDS
            ],
        ]
        for split_seed in SPLIT_SEEDS
    ]
    model_rows = [
        [
            row["model_seed"],
            row["role"],
            f"{fmt(row['test_mean_macro_f1'])} +/- "
            f"{fmt(row['test_sample_std_across_other_factor_macro_f1'])}",
            fmt(row["test_mean_accuracy"]),
            fmt(row["validation_mean_macro_f1"]),
            fmt(row["test_minus_validation_mean_macro_f1"]),
        ]
        for row in model_summary
    ]
    split_rows = [
        [
            row["split_seed"],
            f"{fmt(row['test_mean_macro_f1'])} +/- "
            f"{fmt(row['test_sample_std_across_other_factor_macro_f1'])}",
            fmt(row["test_range_macro_f1"]),
            fmt(row["test_minus_validation_mean_macro_f1"]),
        ]
        for row in split_summary
    ]
    factor_rows = [
        [
            row["source"],
            row["degrees_of_freedom"],
            f"{float(row['sum_of_squares']):.6f}",
            f"{100 * float(row['share_of_total_sum_of_squares']):.1f}%",
        ]
        for row in decomposition["sources"]
    ]
    class_rows = [
        [
            row["class_name"],
            f"{fmt(row['mean_f1_over_model_seed_marginals'])} +/- "
            f"{fmt(row['model_seed_marginal_sample_std_f1'])}",
            fmt(row["mean_precision_over_model_seed_marginals"]),
            fmt(row["mean_recall_over_model_seed_marginals"]),
            fmt(row["model_seed_marginal_range_f1"]),
            f"{float(row['mean_test_windows_per_split']):.1f}",
        ]
        for row in class_summary
    ]
    device_rows = [
        [
            row["group_id"],
            f"{fmt(row['balanced_mean_supported_class_macro_f1'])} +/- "
            f"{fmt(row['model_seed_marginal_sample_std_macro_f1'])}",
            fmt(row["model_seed_marginal_range_macro_f1"]),
            f"{float(row['mean_test_windows_per_cell']):.1f}",
        ]
        for row in device_summary
    ]
    user_rows = [
        [
            row["group_id"],
            f"{fmt(row['balanced_mean_supported_class_macro_f1'])} +/- "
            f"{fmt(row['model_seed_marginal_sample_std_macro_f1'])}",
            fmt(row["model_seed_marginal_range_macro_f1"]),
            f"{float(row['mean_test_windows_per_cell']):.1f}",
        ]
        for row in physical_user_summary
    ]
    confusion_rows = []
    for row in confusion:
        confusion_rows.append(
            [
                row["true_class_name"],
                *[
                    fmt(row[f"predicted_{class_id}_{class_name}"])
                    for class_id, class_name in enumerate(ACTIVITIES)
                ],
            ]
        )
    lines = [
        "# HHAR FedAvg Locked Test V1",
        "",
        "## Boundary",
        "",
        "This is the first and final test evaluation of the frozen HHAR FedAvg "
        "baseline. All 15 pre-registered checkpoints were loaded without retraining "
        "or checkpoint selection. Test results were not used for tuning.",
        "",
        "## Global Test Results",
        "",
        markdown_table(
            ["Split seed", *[f"Model {value}" for value in MODEL_SEEDS]],
            matrix_rows,
        ),
        "",
        markdown_table(
            [
                "Model seed",
                "Role",
                "Test Macro-F1 mean +/- split SD",
                "Test Accuracy",
                "Validation Macro-F1",
                "Test - validation",
            ],
            model_rows,
        ),
        "",
        f"- Balanced test Macro-F1: "
        f"`{fmt(model_variability['balanced_test_mean_macro_f1'])}`.",
        f"- Model-seed marginal SD/range: "
        f"`{fmt(model_variability['marginal_sample_std_macro_f1'])}` / "
        f"`{fmt(model_variability['marginal_range_macro_f1'])}`.",
        f"- Confirmatory four-seed test Macro-F1: "
        f"`{fmt(confirmation['confirmatory_test_mean_of_model_seed_marginals_macro_f1'])} "
        f"+/- {fmt(confirmation['confirmatory_model_seed_marginal_sample_std_macro_f1'])}`.",
        f"- Confirmatory minus development: "
        f"`{fmt(confirmation['confirmatory_minus_development_macro_f1'])}`.",
        "",
        "## Split Variability",
        "",
        markdown_table(
            ["Split seed", "Test mean +/- model SD", "Range", "Test - validation"],
            split_rows,
        ),
        "",
        f"Split-seed marginal SD/range: "
        f"`{fmt(split_variability['marginal_sample_std_macro_f1'])}` / "
        f"`{fmt(split_variability['marginal_range_macro_f1'])}`. This remains "
        "separate from model-seed uncertainty.",
        "",
        "## Descriptive Factor Decomposition",
        "",
        markdown_table(["Source", "df", "SS", "Share of total SS"], factor_rows),
        "",
        "This decomposition is descriptive; no p-values are reported.",
        "",
        "## Per Activity",
        "",
        markdown_table(
            [
                "Activity",
                "F1 mean +/- model SD",
                "Precision",
                "Recall",
                "F1 range",
                "Mean test windows/split",
            ],
            class_rows,
        ),
        "",
        "## Per Device",
        "",
        markdown_table(
            ["Device", "Macro-F1 mean +/- model SD", "Range", "Mean windows/cell"],
            device_rows,
        ),
        "",
        "Device values use supported-class Macro-F1 and are descriptive.",
        "",
        "## Per Physical User",
        "",
        markdown_table(
            ["Physical user", "Macro-F1 mean +/- model SD", "Range", "Mean windows/cell"],
            user_rows,
        ),
        "",
        f"Mean physical-user Macro-F1: `{fmt(user_bootstrap['mean_macro_f1'])}`; "
        f"user-level bootstrap 95% CI "
        f"`[{fmt(user_bootstrap['bootstrap_95pct_ci_lower'])}, "
        f"{fmt(user_bootstrap['bootstrap_95pct_ci_upper'])}]` "
        f"over `{user_bootstrap['n_physical_users']}` physical users.",
        "",
        "User-device pairs are retained only as descriptive client-level metrics; "
        "they are not treated as independent users.",
        "",
        "## Mean Normalized Confusion",
        "",
        markdown_table(
            ["True / predicted", *ACTIVITIES], confusion_rows
        ),
        "",
        "Rows are normalized within each checkpoint and then averaged equally over "
        "all 15 cells.",
        "",
        "## Audit",
        "",
        "Every checkpoint SHA-256, test-window count, prediction row, class support, "
        "confusion total, device set, and physical-user set was validated. All "
        "evaluations were inference-only on `mps`; communication remains the frozen "
        "training cost of `668,913,600` bytes per checkpoint.",
        "",
        "The locked test is now closed. No post-test FedAvg retuning or seed "
        "selection is permitted.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    source_summary, source_rows = source_facts(args)
    split_identity = base.load_split_identity(args)
    args.training_runtime = base.training_runtime(args)
    registered = preregistration(
        args, source_summary, source_rows, split_identity, args.training_runtime
    )
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

    source_cells = source_by_cell(source_rows)
    rows = []
    for index, checkpoint in enumerate(registered["checkpoint_manifest"], start=1):
        cell = (int(checkpoint["model_seed"]), int(checkpoint["split_seed"]))
        rows.append(
            run_cell(
                args,
                checkpoint,
                source_cells[cell],
                split_identity,
                f"locked test {index}/15",
            )
        )
    rows.sort(key=lambda row: (int(row["model_seed"]), int(row["split_seed"])))
    if len(rows) != 15 or len(
        {(int(row["model_seed"]), int(row["split_seed"])) for row in rows}
    ) != 15:
        raise RuntimeError("locked-test result matrix is incomplete")
    if any(bool(row["training_performed"]) for row in rows):
        raise RuntimeError("training occurred during locked-test evaluation")

    model_summary = factor_summary(rows, "model_seed", MODEL_SEEDS)
    split_summary = factor_summary(rows, "split_seed", SPLIT_SEEDS)
    model_variability = marginal_variability(model_summary, "model_seed")
    split_variability = marginal_variability(split_summary, "split_seed")
    confirmation = confirmatory_summary(model_summary)
    decomposition = two_way_decomposition(rows)
    class_cells, class_models, class_summary = collect_class_results(rows)
    device_cells, device_summary = collect_group_results(rows, "device")
    physical_user_cells, physical_user_summary = collect_group_results(
        rows, "physical_user"
    )
    pair_cells = []
    for row in rows:
        for group_row in read_csv(Path(str(row["output_dir"])) / "group_metrics.csv"):
            if group_row["grouping"] == "user_device_pair":
                pair_cells.append(
                    {
                        "model_seed": row["model_seed"],
                        "split_seed": row["split_seed"],
                        **group_row,
                    }
                )
    user_bootstrap = physical_user_bootstrap(physical_user_summary)
    confusion = mean_normalized_confusion(rows)

    write_csv(args.output_root / "cell_results.csv", rows)
    write_csv(args.output_root / "model_seed_summary.csv", model_summary)
    write_csv(args.output_root / "split_seed_summary.csv", split_summary)
    write_csv(args.output_root / "class_cell_results.csv", class_cells)
    write_csv(args.output_root / "class_model_seed_summary.csv", class_models)
    write_csv(args.output_root / "class_summary.csv", class_summary)
    write_csv(args.output_root / "device_cell_results.csv", device_cells)
    write_csv(args.output_root / "device_summary.csv", device_summary)
    write_csv(
        args.output_root / "physical_user_cell_results.csv", physical_user_cells
    )
    write_csv(
        args.output_root / "physical_user_summary.csv", physical_user_summary
    )
    write_csv(args.output_root / "user_device_pair_results.csv", pair_cells)
    write_csv(args.output_root / "mean_normalized_confusion.csv", confusion)
    summary = {
        "status": "locked_test_complete_no_post_test_tuning_permitted",
        "pre_registration": registered,
        "test_access_audit": {
            "first_locked_test": True,
            "all_fifteen_checkpoints_evaluated": True,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "test_driven_tuning_performed": False,
        },
        "frozen_primary_reference": source_summary["frozen_primary_reference"],
        "cell_results": rows,
        "model_seed_summary": model_summary,
        "split_seed_summary": split_summary,
        "model_seed_marginal_test_variability": model_variability,
        "split_seed_marginal_test_variability": split_variability,
        "confirmatory_test_summary": confirmation,
        "two_way_test_decomposition": decomposition,
        "class_summary": class_summary,
        "device_summary": device_summary,
        "physical_user_summary": physical_user_summary,
        "physical_user_bootstrap": user_bootstrap,
        "mean_normalized_confusion": confusion,
        "communication_bytes_per_checkpoint": source_summary[
            "frozen_primary_reference"
        ]["total_communication_bytes"],
        "next_step": (
            "use this frozen test result as the HHAR FedAvg baseline; do not retune "
            "FedAvg from test performance"
        ),
    }
    write_json(args.output_root / "summary.json", summary)
    write_json(args.published_summary_path, summary)
    write_report(
        args.report_path,
        rows,
        model_summary,
        split_summary,
        model_variability,
        split_variability,
        confirmation,
        decomposition,
        class_summary,
        device_summary,
        physical_user_summary,
        user_bootstrap,
        confusion,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
