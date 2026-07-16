#!/usr/bin/env python3
"""Run the validation-only HHAR FedAvg model-seed sensitivity pass."""

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
    "hhar_delivery/reports/hhar/hhar_fedavg_lr_schedule_v1_summary.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/hhar_fedavg_model_seed_sensitivity_v1")
DEFAULT_PREREGISTRATION = Path(
    "hhar_delivery/reports/hhar/"
    "hhar_fedavg_model_seed_sensitivity_v1_preregistration.json"
)
DEFAULT_REPORT = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_model_seed_sensitivity_v1.md"
)
DEFAULT_PUBLISHED_SUMMARY = Path(
    "hhar_delivery/reports/hhar/"
    "hhar_fedavg_model_seed_sensitivity_v1_summary.json"
)
DEVELOPMENT_MODEL_SEED = 20260615
CONFIRMATORY_MODEL_SEEDS = (20260616, 20260617)
MODEL_SEEDS = (DEVELOPMENT_MODEL_SEED, *CONFIRMATORY_MODEL_SEEDS)
TARGET_ROUNDS = 50
LOCAL_EPOCHS = 1
STABILITY_SD_THRESHOLD = 0.05
STABILITY_RANGE_THRESHOLD = 0.10
CONFIG = {
    "config_id": "batchnorm_sgd_lr0p01_constant",
    "model": "1D CNN",
    "norm": "batchnorm",
    "optimizer": "sgd",
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "lr_schedule": "constant",
}


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
) -> tuple[dict[str, object], dict[int, Path]]:
    summary = base.read_json(args.source_summary)
    selected = summary["selected_high_budget"]
    expected = {
        "config_id": CONFIG["config_id"],
        "norm": CONFIG["norm"],
        "optimizer": CONFIG["optimizer"],
        "lr": CONFIG["lr"],
        "momentum": CONFIG["momentum"],
        "local_epochs": LOCAL_EPOCHS,
        "rounds": TARGET_ROUNDS,
    }
    mismatches = [
        f"selected_high_budget.{key}: expected {value!r}, got {selected.get(key)!r}"
        for key, value in expected.items()
        if selected.get(key) != value
    ]
    if summary.get("selection_status") != (
        "hyperparameters_frozen_pending_model_seed_sensitivity"
    ):
        mismatches.append("source selection status is not pending model-seed sensitivity")

    development_rows = [
        row
        for row in summary["candidate_results"]
        if row["config_id"] == CONFIG["config_id"]
        and int(row["model_seed"]) == DEVELOPMENT_MODEL_SEED
    ]
    development_dirs = {
        int(row["split_seed"]): Path(str(row["run_dir"]))
        for row in development_rows
    }
    if tuple(sorted(development_dirs)) != base.SPLIT_SEEDS:
        mismatches.append(
            "source summary does not contain one development run for every split seed"
        )
    for output_dir in development_dirs.values():
        for filename in ("run_config.json", "final_metrics.json"):
            if not (output_dir / filename).exists():
                mismatches.append(f"missing source artifact: {output_dir / filename}")
    if mismatches:
        raise RuntimeError("invalid frozen source:\n- " + "\n- ".join(mismatches))
    return summary, development_dirs


def preregistration(
    args: argparse.Namespace,
    development_dirs: dict[int, Path],
    split_identity: dict[int, dict[str, object]],
    runtime: dict[str, object],
) -> dict[str, object]:
    source_runs = []
    for split_seed in base.SPLIT_SEEDS:
        output_dir = development_dirs[split_seed]
        source_runs.append(
            {
                "split_seed": split_seed,
                "model_seed": DEVELOPMENT_MODEL_SEED,
                "run_dir": str(output_dir),
                "run_config_sha256": base.file_sha256(output_dir / "run_config.json"),
                "final_metrics_sha256": base.file_sha256(
                    output_dir / "final_metrics.json"
                ),
            }
        )
    return {
        "experiment": "hhar_fedavg_model_seed_sensitivity_v1",
        "source_selection_summary": str(args.source_summary),
        "source_selection_summary_sha256": base.file_sha256(args.source_summary),
        "source_selection_result_commit": source_commit(args.source_summary),
        "trainer": str(base.TRAINER),
        "trainer_sha256": base.file_sha256(base.TRAINER),
        "training_runtime": runtime,
        "split_identity": [
            split_identity[split_seed] for split_seed in base.SPLIT_SEEDS
        ],
        "frozen_configuration": {
            **CONFIG,
            "rounds": TARGET_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": args.batch_size,
            "client_fraction": args.client_fraction,
            "eval_every": args.eval_every,
        },
        "crossed_design": {
            "split_seeds": list(base.SPLIT_SEEDS),
            "model_seeds": list(MODEL_SEEDS),
            "development_model_seed": DEVELOPMENT_MODEL_SEED,
            "confirmatory_model_seeds": list(CONFIRMATORY_MODEL_SEEDS),
            "total_cells": len(base.SPLIT_SEEDS) * len(MODEL_SEEDS),
            "reused_development_cells": len(base.SPLIT_SEEDS),
            "new_confirmatory_cells": len(base.SPLIT_SEEDS)
            * len(CONFIRMATORY_MODEL_SEEDS),
            "reused_development_runs": source_runs,
        },
        "selection_boundary": {
            "evaluation_splits": list(base.EVALUATION_SPLITS),
            "test_metrics_generated": False,
            "selection_or_retuning_performed": False,
            "primary_metric": "validation global Macro-F1",
            "pair_level_metrics_used_for_decision": False,
        },
        "analysis_plan": {
            "primary_confirmatory_estimate": (
                "Mean of the two confirmatory model-seed marginal validation "
                "Macro-F1 values, where each marginal first averages the three "
                "frozen split seeds."
            ),
            "model_seed_variability": (
                "Sample SD and range across the three model-seed marginal means."
            ),
            "split_variability": (
                "Reported separately as sample SD and range across the three "
                "split-seed marginal means."
            ),
            "paired_check": (
                "For each confirmatory model seed, subtract the development-seed "
                "score within each identical split before averaging."
            ),
            "factor_decomposition": (
                "Descriptive balanced two-way decomposition into split, model-seed, "
                "and split-by-model interaction sums of squares; no p-values."
            ),
            "prohibited_summary": (
                "Do not use a naive pooled 9-cell standard deviation as either "
                "split or model-seed uncertainty."
            ),
        },
        "stability_rule": {
            "applied_to": "three model-seed marginal validation Macro-F1 means",
            "sample_std_max": STABILITY_SD_THRESHOLD,
            "range_max": STABILITY_RANGE_THRESHOLD,
            "threshold_provenance": (
                "Same absolute SD and range thresholds pre-registered for HHAR "
                "split-seed sensitivity V1."
            ),
            "pass_requires": "both thresholds",
            "if_pass": (
                "Freeze the baseline as ready for a separately pre-registered "
                "locked test evaluation."
            ),
            "if_fail": (
                "Keep hyperparameters frozen, do not inspect test metrics, and "
                "expand model-seed characterization to five seeds before test."
            ),
            "if_fail_additional_model_seeds": [20260618, 20260619],
        },
        "stop_rule": (
            "This pass cannot change FedAvg hyperparameters, schedules, rounds, or "
            "budgets. Stop after the pre-registered 3x3 matrix and report the result."
        ),
    }


def new_run_dir(args: argparse.Namespace, model_seed: int, split_seed: int) -> Path:
    return (
        args.output_root
        / f"model_seed{model_seed}"
        / f"split_seed{split_seed}"
    )


def expected_run_config(
    args: argparse.Namespace, model_seed: int, split_seed: int
) -> dict[str, object]:
    return {
        "dataset": "HHAR",
        "manifest_version": "V1",
        "manifest_dir": str(base.manifest_dir(args, split_seed)),
        "archive": str(args.archive),
        "cache_dir": str(args.cache_dir),
        "rounds": TARGET_ROUNDS,
        "client_fraction": args.client_fraction,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": args.batch_size,
        "lr": CONFIG["lr"],
        "momentum": CONFIG["momentum"],
        "weight_decay": CONFIG["weight_decay"],
        "optimizer": CONFIG["optimizer"],
        "norm": CONFIG["norm"],
        "groupnorm_groups": 8,
        "eval_every": args.eval_every,
        "evaluation_splits": list(base.EVALUATION_SPLITS),
        "seed": model_seed,
        "python_version": args.training_runtime["python_version"],
        "torch_version": args.training_runtime["torch_version"],
    }


def validate_run(
    args: argparse.Namespace,
    output_dir: Path,
    model_seed: int,
    split_seed: int,
    split_identity: dict[int, dict[str, object]],
    expected_communication: int,
) -> dict[str, object]:
    run_config = base.read_json(output_dir / "run_config.json")
    expected = expected_run_config(args, model_seed, split_seed)
    mismatches = [
        f"run_config.{key}: expected {value!r}, got {run_config.get(key)!r}"
        for key, value in expected.items()
        if run_config.get(key) != value
    ]
    if args.device != "auto" and run_config.get("resolved_device") != args.device:
        mismatches.append("resolved device does not match requested device")

    manifest_sanity = base.read_json(output_dir / "manifest_sanity.json")
    if manifest_sanity["split_windows"] != split_identity[split_seed]["split_windows"]:
        mismatches.append("manifest split counts do not match frozen split identity")

    final = base.read_json(output_dir / "final_metrics.json")
    history = base.read_json(output_dir / "metrics_history.json")
    expected_splits = set(base.EVALUATION_SPLITS)
    if int(final["round"]) != TARGET_ROUNDS:
        mismatches.append(f"final round is {final['round']}, expected {TARGET_ROUNDS}")
    if set(final["metrics"]) != expected_splits:
        mismatches.append("final metrics contain an unregistered evaluation split")
    if any(set(record["metrics"]) != expected_splits for record in history):
        mismatches.append("metrics history contains an unregistered evaluation split")
    expected_rounds = [0, *range(args.eval_every, TARGET_ROUNDS + 1, args.eval_every)]
    observed_rounds = [int(record["round"]) for record in history]
    if observed_rounds != expected_rounds:
        mismatches.append(
            f"checkpoint rounds are {observed_rounds}, expected {expected_rounds}"
        )

    round_rows = base.read_csv(output_dir / "round_metrics.csv")
    if any(key.startswith("test_") for key in round_rows[0]):
        mismatches.append("round_metrics.csv contains test metrics")
    group_rows = base.read_csv(output_dir / "group_metrics.csv")
    if {row["split"] for row in group_rows} != expected_splits:
        mismatches.append("group_metrics.csv contains an unregistered split")
    communication = int(final["communication"]["total_bytes"])
    if communication != expected_communication:
        mismatches.append(
            f"communication is {communication}, expected {expected_communication}"
        )
    if mismatches:
        raise RuntimeError(
            f"invalid model-seed output at {output_dir}:\n- "
            + "\n- ".join(mismatches)
        )

    validation = final["metrics"]["validation"]
    return {
        "split_seed": split_seed,
        "model_seed": model_seed,
        "model_seed_role": (
            "development" if model_seed == DEVELOPMENT_MODEL_SEED else "confirmatory"
        ),
        "validation_accuracy": validation["accuracy"],
        "validation_macro_f1": validation["macro_f1"],
        "validation_pair_mean_macro_f1_descriptive": validation["per_user"][
            "mean_macro_f1"
        ],
        "rounds": TARGET_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "total_communication_bytes": communication,
        "resolved_device": run_config["resolved_device"],
        "test_metrics_generated": False,
        "run_dir": str(output_dir),
    }


def run_one(
    args: argparse.Namespace,
    model_seed: int,
    split_seed: int,
    split_identity: dict[int, dict[str, object]],
    expected_communication: int,
    ordinal: str,
) -> dict[str, object]:
    output_dir = new_run_dir(args, model_seed, split_seed)
    if (output_dir / "final_metrics.json").exists() and not args.force:
        row = validate_run(
            args,
            output_dir,
            model_seed,
            split_seed,
            split_identity,
            expected_communication,
        )
        print(f"[{ordinal}] validated existing {output_dir}", flush=True)
        return row

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.training_python),
        str(base.TRAINER),
        "--manifest-dir",
        str(base.manifest_dir(args, split_seed)),
        "--archive",
        str(args.archive),
        "--cache-dir",
        str(args.cache_dir),
        "--output-dir",
        str(output_dir),
        "--rounds",
        str(TARGET_ROUNDS),
        "--eval-every",
        str(args.eval_every),
        "--client-fraction",
        str(args.client_fraction),
        "--local-epochs",
        str(LOCAL_EPOCHS),
        "--batch-size",
        str(args.batch_size),
        "--optimizer",
        str(CONFIG["optimizer"]),
        "--lr",
        str(CONFIG["lr"]),
        "--momentum",
        str(CONFIG["momentum"]),
        "--norm",
        str(CONFIG["norm"]),
        "--evaluation-splits",
        *base.EVALUATION_SPLITS,
        "--device",
        args.device,
        "--seed",
        str(model_seed),
    ]
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
        model_seed,
        split_seed,
        split_identity,
        expected_communication,
    )


def marginal_summary(
    rows: list[dict[str, object]], key: str, values: tuple[int, ...]
) -> list[dict[str, object]]:
    other_key = "split_seed" if key == "model_seed" else "model_seed"
    expected_other = len(base.SPLIT_SEEDS) if key == "model_seed" else len(MODEL_SEEDS)
    output = []
    for value in values:
        selected = [row for row in rows if int(row[key]) == value]
        scores = [float(row["validation_macro_f1"]) for row in selected]
        if len(scores) != expected_other or len({int(row[other_key]) for row in selected}) != expected_other:
            raise RuntimeError(f"incomplete crossed design for {key}={value}")
        output.append(
            {
                key: value,
                "role": (
                    "development"
                    if key == "model_seed" and value == DEVELOPMENT_MODEL_SEED
                    else "confirmatory"
                    if key == "model_seed"
                    else "frozen_split"
                ),
                "n_cells": len(scores),
                "validation_mean_macro_f1": statistics.mean(scores),
                "validation_sample_std_across_other_factor": base.sample_std(scores),
                "validation_min_macro_f1": min(scores),
                "validation_max_macro_f1": max(scores),
                "validation_range_macro_f1": max(scores) - min(scores),
            }
        )
    return output


def marginal_stability(
    rows: list[dict[str, object]], factor: str
) -> dict[str, object]:
    values = [float(row["validation_mean_macro_f1"]) for row in rows]
    sample_sd = base.sample_std(values)
    value_range = max(values) - min(values)
    result = {
        "factor": factor,
        "n_marginal_means": len(values),
        "balanced_grand_mean_macro_f1": statistics.mean(values),
        "marginal_sample_std_macro_f1": sample_sd,
        "marginal_min_macro_f1": min(values),
        "marginal_max_macro_f1": max(values),
        "marginal_range_macro_f1": value_range,
    }
    if factor == "model_seed":
        result.update(
            {
                "sample_std_threshold": STABILITY_SD_THRESHOLD,
                "range_threshold": STABILITY_RANGE_THRESHOLD,
                "sample_std_pass": sample_sd <= STABILITY_SD_THRESHOLD,
                "range_pass": value_range <= STABILITY_RANGE_THRESHOLD,
                "stable": sample_sd <= STABILITY_SD_THRESHOLD
                and value_range <= STABILITY_RANGE_THRESHOLD,
            }
        )
    return result


def confirmatory_summary(
    model_rows: list[dict[str, object]], rows: list[dict[str, object]]
) -> dict[str, object]:
    by_seed = {int(row["model_seed"]): row for row in model_rows}
    development = float(
        by_seed[DEVELOPMENT_MODEL_SEED]["validation_mean_macro_f1"]
    )
    confirmatory_marginals = [
        float(by_seed[seed]["validation_mean_macro_f1"])
        for seed in CONFIRMATORY_MODEL_SEEDS
    ]
    confirmatory_cells = [
        float(row["validation_macro_f1"])
        for row in rows
        if int(row["model_seed"]) in CONFIRMATORY_MODEL_SEEDS
    ]
    confirmatory_mean = statistics.mean(confirmatory_marginals)
    return {
        "development_model_seed": DEVELOPMENT_MODEL_SEED,
        "development_marginal_mean_macro_f1": development,
        "confirmatory_model_seeds": list(CONFIRMATORY_MODEL_SEEDS),
        "n_confirmatory_cells": len(confirmatory_cells),
        "confirmatory_mean_of_model_seed_marginals_macro_f1": confirmatory_mean,
        "confirmatory_model_seed_marginal_sample_std_macro_f1": base.sample_std(
            confirmatory_marginals
        ),
        "confirmatory_min_model_seed_marginal_macro_f1": min(
            confirmatory_marginals
        ),
        "confirmatory_max_model_seed_marginal_macro_f1": max(
            confirmatory_marginals
        ),
        "confirmatory_minus_development_macro_f1": confirmatory_mean - development,
    }


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    scores = {
        (int(row["model_seed"]), int(row["split_seed"])): float(
            row["validation_macro_f1"]
        )
        for row in rows
    }
    output = []
    for model_seed in CONFIRMATORY_MODEL_SEEDS:
        deltas = [
            scores[(model_seed, split_seed)]
            - scores[(DEVELOPMENT_MODEL_SEED, split_seed)]
            for split_seed in base.SPLIT_SEEDS
        ]
        output.append(
            {
                "model_seed": model_seed,
                "reference_model_seed": DEVELOPMENT_MODEL_SEED,
                "split_seeds": list(base.SPLIT_SEEDS),
                "paired_deltas_by_split": deltas,
                "mean_paired_delta_macro_f1": statistics.mean(deltas),
                "sample_std_paired_delta_macro_f1": base.sample_std(deltas),
                "min_paired_delta_macro_f1": min(deltas),
                "max_paired_delta_macro_f1": max(deltas),
            }
        )
    return output


def two_way_decomposition(rows: list[dict[str, object]]) -> dict[str, object]:
    scores = {
        (int(row["split_seed"]), int(row["model_seed"])): float(
            row["validation_macro_f1"]
        )
        for row in rows
    }
    if len(scores) != len(base.SPLIT_SEEDS) * len(MODEL_SEEDS):
        raise RuntimeError("two-way decomposition requires the complete 3x3 matrix")
    all_values = list(scores.values())
    grand = statistics.mean(all_values)
    split_means = {
        split_seed: statistics.mean(
            [scores[(split_seed, model_seed)] for model_seed in MODEL_SEEDS]
        )
        for split_seed in base.SPLIT_SEEDS
    }
    model_means = {
        model_seed: statistics.mean(
            [scores[(split_seed, model_seed)] for split_seed in base.SPLIT_SEEDS]
        )
        for model_seed in MODEL_SEEDS
    }
    ss_total = sum((value - grand) ** 2 for value in all_values)
    ss_split = len(MODEL_SEEDS) * sum(
        (value - grand) ** 2 for value in split_means.values()
    )
    ss_model = len(base.SPLIT_SEEDS) * sum(
        (value - grand) ** 2 for value in model_means.values()
    )
    ss_interaction = max(ss_total - ss_split - ss_model, 0.0)
    components = [
        ("split_seed", len(base.SPLIT_SEEDS) - 1, ss_split),
        ("model_seed", len(MODEL_SEEDS) - 1, ss_model),
        (
            "split_by_model_seed_interaction",
            (len(base.SPLIT_SEEDS) - 1) * (len(MODEL_SEEDS) - 1),
            ss_interaction,
        ),
    ]
    sources = [
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
    ]
    by_source = {row["source"]: row for row in sources}
    interaction_ms = float(
        by_source["split_by_model_seed_interaction"]["mean_square"]
    )
    return {
        "method": (
            "balanced two-way descriptive decomposition without replicated cells; "
            "interaction is the residual and no inferential test is performed"
        ),
        "balanced_grand_mean_macro_f1": grand,
        "sources": sources,
        "method_of_moments_variance_components_descriptive": {
            "split_seed": max(
                (float(by_source["split_seed"]["mean_square"]) - interaction_ms)
                / len(MODEL_SEEDS),
                0.0,
            ),
            "model_seed": max(
                (float(by_source["model_seed"]["mean_square"]) - interaction_ms)
                / len(base.SPLIT_SEEDS),
                0.0,
            ),
            "split_by_model_seed_interaction": interaction_ms,
        },
    }


def validation_device_model_seed_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_device_seed: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        model_seed = int(row["model_seed"])
        group_rows = base.read_csv(Path(str(row["run_dir"])) / "group_metrics.csv")
        for group_row in group_rows:
            if group_row["split"] != "validation" or group_row["grouping"] != "device":
                continue
            key = (group_row["group_id"], model_seed)
            by_device_seed.setdefault(key, []).append(float(group_row["macro_f1"]))

    devices = sorted({key[0] for key in by_device_seed})
    output = []
    for device_id in devices:
        model_marginals = []
        for model_seed in MODEL_SEEDS:
            values = by_device_seed.get((device_id, model_seed), [])
            if len(values) != len(base.SPLIT_SEEDS):
                raise RuntimeError(
                    f"incomplete device summary for {device_id}, seed {model_seed}"
                )
            model_marginals.append(statistics.mean(values))
        sample_sd = base.sample_std(model_marginals)
        value_range = max(model_marginals) - min(model_marginals)
        output.append(
            {
                "device_id": device_id,
                "validation_mean_of_model_seed_marginals_macro_f1": statistics.mean(
                    model_marginals
                ),
                "validation_model_seed_marginal_sample_std_macro_f1": sample_sd,
                "validation_model_seed_marginal_min_macro_f1": min(model_marginals),
                "validation_model_seed_marginal_max_macro_f1": max(model_marginals),
                "validation_model_seed_marginal_range_macro_f1": value_range,
                "descriptive_stable_at_global_thresholds": sample_sd
                <= STABILITY_SD_THRESHOLD
                and value_range <= STABILITY_RANGE_THRESHOLD,
            }
        )
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
    prereg: dict[str, object],
    rows: list[dict[str, object]],
    model_summary: list[dict[str, object]],
    split_summary: list[dict[str, object]],
    model_stability: dict[str, object],
    split_stability: dict[str, object],
    confirmation: dict[str, object],
    deltas: list[dict[str, object]],
    decomposition: dict[str, object],
    devices: list[dict[str, object]],
    status: str,
) -> None:
    score_by_cell = {
        (int(row["split_seed"]), int(row["model_seed"])): row[
            "validation_macro_f1"
        ]
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
        for split_seed in base.SPLIT_SEEDS
    ]
    model_rows = [
        [
            row["model_seed"],
            row["role"],
            fmt(row["validation_mean_macro_f1"]),
            fmt(row["validation_sample_std_across_other_factor"]),
            fmt(row["validation_range_macro_f1"]),
        ]
        for row in model_summary
    ]
    split_rows = [
        [
            row["split_seed"],
            fmt(row["validation_mean_macro_f1"]),
            fmt(row["validation_sample_std_across_other_factor"]),
            fmt(row["validation_range_macro_f1"]),
        ]
        for row in split_summary
    ]
    delta_rows = [
        [
            row["model_seed"],
            *[fmt(value) for value in row["paired_deltas_by_split"]],
            fmt(row["mean_paired_delta_macro_f1"]),
        ]
        for row in deltas
    ]
    decomposition_rows = [
        [
            row["source"],
            row["degrees_of_freedom"],
            f"{float(row['sum_of_squares']):.6f}",
            f"{float(row['mean_square']):.6f}",
            f"{100 * float(row['share_of_total_sum_of_squares']):.1f}%",
        ]
        for row in decomposition["sources"]
    ]
    device_rows = [
        [
            row["device_id"],
            f"{fmt(row['validation_mean_of_model_seed_marginals_macro_f1'])} +/- "
            f"{fmt(row['validation_model_seed_marginal_sample_std_macro_f1'])}",
            fmt(row["validation_model_seed_marginal_range_macro_f1"]),
            row["descriptive_stable_at_global_thresholds"],
        ]
        for row in devices
    ]
    ready = bool(model_stability["stable"])
    lines = [
        "# HHAR FedAvg Model-Seed Sensitivity V1",
        "",
        "## Purpose",
        "",
        "This validation-only pass measures optimization-seed sensitivity after "
        "the HHAR FedAvg configuration was frozen. It does not select or retune "
        "hyperparameters. Model seed `20260615` is the development seed used during "
        "tuning; seeds `20260616` and `20260617` are confirmatory.",
        "",
        "## Pre-Registered Protocol",
        "",
        f"- Split seeds: `{list(base.SPLIT_SEEDS)}`.",
        f"- Model seeds: `{list(MODEL_SEEDS)}`; full `3 x 3` crossed design.",
        "- Frozen setting: BatchNorm, SGD momentum `0.9`, constant learning rate "
        "`0.01`, 1 local epoch, 50 rounds, full participation.",
        "- Primary metric: validation global Macro-F1; test metrics were not generated.",
        "- Stability rule: sample SD across model-seed marginal means <= "
        f"`{STABILITY_SD_THRESHOLD}` and range <= `{STABILITY_RANGE_THRESHOLD}`.",
        "- The two confirmatory model seeds are summarized separately because the "
        "development seed contributed to hyperparameter selection.",
        "",
        "## Crossed Results",
        "",
        markdown_table(
            ["Split seed", *[f"Model {seed}" for seed in MODEL_SEEDS]], matrix_rows
        ),
        "",
        "## Model-Seed Variability",
        "",
        markdown_table(
            ["Model seed", "Role", "Val mean", "Across-split SD", "Across-split range"],
            model_rows,
        ),
        "",
        "Across the three model-seed marginal means:",
        "",
        f"- Balanced grand mean: `{fmt(model_stability['balanced_grand_mean_macro_f1'])}`.",
        f"- Model-seed marginal sample SD: "
        f"`{fmt(model_stability['marginal_sample_std_macro_f1'])}`.",
        f"- Model-seed marginal range: "
        f"`{fmt(model_stability['marginal_range_macro_f1'])}`.",
        f"- Pre-registered stability pass: `{model_stability['stable']}`.",
        "",
        "Confirmatory estimate:",
        "",
        f"- Development-seed marginal mean: "
        f"`{fmt(confirmation['development_marginal_mean_macro_f1'])}`.",
        f"- Held-out model-seed marginal mean: "
        f"`{fmt(confirmation['confirmatory_mean_of_model_seed_marginals_macro_f1'])}`.",
        f"- Confirmatory minus development: "
        f"`{fmt(confirmation['confirmatory_minus_development_macro_f1'])}`.",
        "",
        "## Paired Seed Differences",
        "",
        "Each difference uses the same split under the development model seed as "
        "its reference.",
        "",
        markdown_table(
            [
                "Confirmatory model seed",
                *[f"Split {seed}" for seed in base.SPLIT_SEEDS],
                "Mean delta",
            ],
            delta_rows,
        ),
        "",
        "## Split Variability",
        "",
        markdown_table(
            ["Split seed", "Val mean", "Across-model SD", "Across-model range"],
            split_rows,
        ),
        "",
        f"Split-seed marginal sample SD: "
        f"`{fmt(split_stability['marginal_sample_std_macro_f1'])}`; range: "
        f"`{fmt(split_stability['marginal_range_macro_f1'])}`. These are reported "
        "separately from model-seed variability.",
        "",
        "## Descriptive Factor Decomposition",
        "",
        markdown_table(
            ["Source", "df", "SS", "MS", "Share of total SS"],
            decomposition_rows,
        ),
        "",
        "This balanced two-way decomposition is descriptive. With one deterministic "
        "run per cell, the residual is the split-by-model-seed interaction; no "
        "p-values or formal variance claims are made.",
        "",
        "## Validation By Device",
        "",
        "Each device value first averages the three splits within each model seed; "
        "the displayed SD and range are then computed over the three model-seed "
        "marginals. Device stability is descriptive and does not gate the global decision.",
        "",
        markdown_table(
            ["Device", "Val mean +/- model-seed SD", "Range", "Descriptive stable"],
            device_rows,
        ),
        "",
        "## Decision",
        "",
        f"Status: `{status}`.",
        "",
        (
            "The frozen baseline passed the pre-registered model-seed stability "
            "rule and is ready for a separately pre-registered locked test evaluation."
            if ready
            else "The frozen baseline failed the pre-registered stability rule. "
            "Hyperparameters remain frozen, test remains locked, and model-seed "
            "characterization must expand to five seeds."
        ),
        "",
        "No naive pooled 9-cell standard deviation is reported, because it would "
        "conflate split and model-seed uncertainty.",
        "",
        "## Audit",
        "",
        "All nine runs were validated against the frozen split identities, runtime, "
        "configuration, checkpoint schedule, and communication budget. Every metric "
        "artifact contains only `train` and `validation` performance.",
        "",
        f"Pre-registration source: `{prereg['source_selection_result_commit']}`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    source_summary, development_dirs = source_facts(args)
    split_identity = base.load_split_identity(args)
    args.training_runtime = base.training_runtime(args)
    registered = preregistration(
        args, development_dirs, split_identity, args.training_runtime
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

    expected_communication = int(
        source_summary["selected_high_budget"]["total_communication_bytes"]
    )
    rows = []
    for split_seed in base.SPLIT_SEEDS:
        rows.append(
            validate_run(
                args,
                development_dirs[split_seed],
                DEVELOPMENT_MODEL_SEED,
                split_seed,
                split_identity,
                expected_communication,
            )
        )
    print("validated 3 reused development-seed cells", flush=True)

    ordinal = 0
    total_new = len(base.SPLIT_SEEDS) * len(CONFIRMATORY_MODEL_SEEDS)
    for model_seed in CONFIRMATORY_MODEL_SEEDS:
        for split_seed in base.SPLIT_SEEDS:
            ordinal += 1
            rows.append(
                run_one(
                    args,
                    model_seed,
                    split_seed,
                    split_identity,
                    expected_communication,
                    f"confirmatory {ordinal}/{total_new}",
                )
            )
    rows.sort(key=lambda row: (int(row["model_seed"]), int(row["split_seed"])))
    if len(rows) != 9 or len(
        {(int(row["model_seed"]), int(row["split_seed"])) for row in rows}
    ) != 9:
        raise RuntimeError("model-seed result matrix is incomplete")
    if any(bool(row["test_metrics_generated"]) for row in rows):
        raise RuntimeError("test metric leakage detected")
    devices_used = {str(row["resolved_device"]) for row in rows}
    if len(devices_used) != 1:
        raise RuntimeError(f"resolved device changed across runs: {devices_used}")

    model_summary = marginal_summary(rows, "model_seed", MODEL_SEEDS)
    split_summary = marginal_summary(rows, "split_seed", base.SPLIT_SEEDS)
    model_stability = marginal_stability(model_summary, "model_seed")
    split_stability = marginal_stability(split_summary, "split_seed")
    confirmation = confirmatory_summary(model_summary, rows)
    deltas = paired_deltas(rows)
    decomposition = two_way_decomposition(rows)
    devices = validation_device_model_seed_summary(rows)
    stable = bool(model_stability["stable"])
    status = (
        "model_seed_sensitivity_complete_ready_for_locked_test"
        if stable
        else "model_seed_instability_requires_five_seed_expansion"
    )

    base.write_csv(args.output_root / "run_results.csv", rows)
    base.write_csv(args.output_root / "model_seed_summary.csv", model_summary)
    base.write_csv(args.output_root / "split_seed_summary.csv", split_summary)
    base.write_csv(args.output_root / "paired_deltas.csv", deltas)
    base.write_csv(
        args.output_root / "validation_device_model_seed_summary.csv", devices
    )
    summary = {
        "status": status,
        "pre_registration": registered,
        "test_access_audit": {
            "evaluation_splits": list(base.EVALUATION_SPLITS),
            "test_metrics_generated": False,
            "all_nine_runs_validated": True,
        },
        "frozen_primary_reference": {
            **CONFIG,
            "rounds": TARGET_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "total_communication_bytes": expected_communication,
            "resolved_device": next(iter(devices_used)),
        },
        "run_results": rows,
        "model_seed_summary": model_summary,
        "split_seed_summary": split_summary,
        "model_seed_marginal_stability": model_stability,
        "split_seed_marginal_stability": split_stability,
        "confirmatory_summary": confirmation,
        "paired_model_seed_deltas": deltas,
        "two_way_decomposition": decomposition,
        "validation_device_model_seed_summary": devices,
        "next_step": (
            "separately pre-register and run the first locked test evaluation"
            if stable
            else "expand the frozen configuration to five model seeds before test"
        ),
    }
    base.write_json(args.output_root / "summary.json", summary)
    base.write_json(args.published_summary_path, summary)
    write_report(
        args.report_path,
        registered,
        rows,
        model_summary,
        split_summary,
        model_stability,
        split_stability,
        confirmation,
        deltas,
        decomposition,
        devices,
        status,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
