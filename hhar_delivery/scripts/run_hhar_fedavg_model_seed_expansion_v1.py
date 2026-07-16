#!/usr/bin/env python3
"""Expand frozen HHAR FedAvg model-seed characterization from three to five."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIOR_SCRIPT = (
    REPO_ROOT
    / "hhar_delivery/scripts/run_hhar_fedavg_model_seed_sensitivity_v1.py"
)
SPEC = importlib.util.spec_from_file_location("hhar_fedavg_model_seed_v1", PRIOR_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load prior runner at {PRIOR_SCRIPT}")
prior = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prior
SPEC.loader.exec_module(prior)
base = prior.base


DEFAULT_SOURCE_SUMMARY = Path(
    "hhar_delivery/reports/hhar/"
    "hhar_fedavg_model_seed_sensitivity_v1_summary.json"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/hhar_fedavg_model_seed_expansion_v1")
DEFAULT_PREREGISTRATION = Path(
    "hhar_delivery/reports/hhar/"
    "hhar_fedavg_model_seed_expansion_v1_preregistration.json"
)
DEFAULT_REPORT = Path(
    "hhar_delivery/reports/hhar/hhar_fedavg_model_seed_expansion_v1.md"
)
DEFAULT_PUBLISHED_SUMMARY = Path(
    "hhar_delivery/reports/hhar/"
    "hhar_fedavg_model_seed_expansion_v1_summary.json"
)
DEVELOPMENT_MODEL_SEED = prior.DEVELOPMENT_MODEL_SEED
EXISTING_MODEL_SEEDS = prior.MODEL_SEEDS
ADDITIONAL_MODEL_SEEDS = (20260618, 20260619)
CONFIRMATORY_MODEL_SEEDS = (*prior.CONFIRMATORY_MODEL_SEEDS, *ADDITIONAL_MODEL_SEEDS)
MODEL_SEEDS = (DEVELOPMENT_MODEL_SEED, *CONFIRMATORY_MODEL_SEEDS)


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


def source_facts(args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary = base.read_json(args.source_summary)
    mismatches = []
    if summary.get("status") != (
        "model_seed_instability_requires_five_seed_expansion"
    ):
        mismatches.append("source status does not trigger the five-seed expansion")
    stability = summary.get("model_seed_marginal_stability", {})
    if stability.get("stable") is not False:
        mismatches.append("source model-seed stability result is not false")
    if float(stability.get("marginal_sample_std_macro_f1", 0.0)) <= float(
        stability.get("sample_std_threshold", prior.STABILITY_SD_THRESHOLD)
    ):
        mismatches.append("source sample SD does not exceed its threshold")
    registered_extra = summary["pre_registration"]["stability_rule"].get(
        "if_fail_additional_model_seeds"
    )
    if registered_extra != list(ADDITIONAL_MODEL_SEEDS):
        mismatches.append("source pre-registration names different expansion seeds")
    rows = list(summary.get("run_results", []))
    cells = {(int(row["model_seed"]), int(row["split_seed"])) for row in rows}
    expected_cells = {
        (model_seed, split_seed)
        for model_seed in EXISTING_MODEL_SEEDS
        for split_seed in base.SPLIT_SEEDS
    }
    if cells != expected_cells:
        mismatches.append("source summary is not the complete registered 3x3 matrix")
    for row in rows:
        output_dir = Path(str(row["run_dir"]))
        for filename in (
            "run_config.json",
            "final_metrics.json",
            "metrics_history.json",
            "group_metrics.csv",
            "final_model.pt",
        ):
            if not (output_dir / filename).exists():
                mismatches.append(f"missing source artifact: {output_dir / filename}")
    if mismatches:
        raise RuntimeError("invalid expansion source:\n- " + "\n- ".join(mismatches))
    return summary, rows


def source_cell_identity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    identities = []
    for row in sorted(
        rows, key=lambda item: (int(item["model_seed"]), int(item["split_seed"]))
    ):
        output_dir = Path(str(row["run_dir"]))
        identities.append(
            {
                "model_seed": int(row["model_seed"]),
                "split_seed": int(row["split_seed"]),
                "run_dir": str(output_dir),
                "run_config_sha256": base.file_sha256(output_dir / "run_config.json"),
                "final_metrics_sha256": base.file_sha256(
                    output_dir / "final_metrics.json"
                ),
                "final_model_sha256": base.file_sha256(output_dir / "final_model.pt"),
            }
        )
    return identities


def preregistration(
    args: argparse.Namespace,
    source_summary: dict[str, object],
    source_rows: list[dict[str, object]],
    split_identity: dict[int, dict[str, object]],
    runtime: dict[str, object],
) -> dict[str, object]:
    trigger = source_summary["model_seed_marginal_stability"]
    return {
        "experiment": "hhar_fedavg_model_seed_expansion_v1",
        "source_sensitivity_summary": str(args.source_summary),
        "source_sensitivity_summary_sha256": base.file_sha256(args.source_summary),
        "source_sensitivity_result_commit": source_commit(args.source_summary),
        "expansion_trigger": {
            "model_seed_marginal_sample_std_macro_f1": trigger[
                "marginal_sample_std_macro_f1"
            ],
            "sample_std_threshold": trigger["sample_std_threshold"],
            "sample_std_pass": trigger["sample_std_pass"],
            "model_seed_marginal_range_macro_f1": trigger[
                "marginal_range_macro_f1"
            ],
            "range_threshold": trigger["range_threshold"],
            "range_pass": trigger["range_pass"],
        },
        "trainer": str(base.TRAINER),
        "trainer_sha256": base.file_sha256(base.TRAINER),
        "training_runtime": runtime,
        "split_identity": [
            split_identity[split_seed] for split_seed in base.SPLIT_SEEDS
        ],
        "frozen_configuration": {
            **prior.CONFIG,
            "rounds": prior.TARGET_ROUNDS,
            "local_epochs": prior.LOCAL_EPOCHS,
            "batch_size": args.batch_size,
            "client_fraction": args.client_fraction,
            "eval_every": args.eval_every,
        },
        "crossed_design": {
            "split_seeds": list(base.SPLIT_SEEDS),
            "model_seeds": list(MODEL_SEEDS),
            "development_model_seed": DEVELOPMENT_MODEL_SEED,
            "confirmatory_model_seeds": list(CONFIRMATORY_MODEL_SEEDS),
            "additional_model_seeds": list(ADDITIONAL_MODEL_SEEDS),
            "total_cells": len(base.SPLIT_SEEDS) * len(MODEL_SEEDS),
            "reused_cells": len(source_rows),
            "new_cells": len(base.SPLIT_SEEDS) * len(ADDITIONAL_MODEL_SEEDS),
            "reused_cell_identity": source_cell_identity(source_rows),
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
                "Mean of four confirmatory model-seed marginal validation Macro-F1 "
                "values; each marginal averages the three frozen split seeds first."
            ),
            "model_seed_variability": (
                "Sample SD and range across five model-seed marginal means."
            ),
            "split_variability": (
                "Report separately as sample SD and range across three split-seed "
                "marginal means, each averaged over five model seeds."
            ),
            "paired_check": (
                "Subtract the development-seed score within each identical split "
                "for every confirmatory model seed."
            ),
            "factor_decomposition": (
                "Descriptive balanced 3x5 two-way decomposition into split, model "
                "seed, and interaction sums of squares; no p-values."
            ),
            "prohibited_summary": (
                "Do not use a naive pooled 15-cell standard deviation as either "
                "split or model-seed uncertainty."
            ),
        },
        "stability_rule": {
            "applied_to": "five model-seed marginal validation Macro-F1 means",
            "sample_std_max": prior.STABILITY_SD_THRESHOLD,
            "range_max": prior.STABILITY_RANGE_THRESHOLD,
            "pass_requires": "both thresholds",
            "interpretation_if_pass": "model-seed-stable frozen baseline",
            "interpretation_if_fail": (
                "model-seed-sensitive frozen baseline; retain the full five-seed "
                "distribution in every headline comparison"
            ),
        },
        "locked_test_policy_after_expansion": {
            "test_remains_locked_during_this_pass": True,
            "separate_preregistration_required": True,
            "evaluation_only": True,
            "retraining_prohibited": True,
            "all_fifteen_final_checkpoints_must_be_evaluated": True,
            "seed_selection_prohibited": True,
            "uncertainty_reporting": (
                "Report model-seed marginals and split-seed marginals separately."
            ),
        },
        "stop_rule": (
            "After completing the 3x5 matrix, do not add model seeds or retune the "
            "FedAvg configuration. Record stable or seed-sensitive status, then "
            "proceed to a separately pre-registered evaluation-only locked test."
        ),
    }


def aggregate_by_factor(
    rows: list[dict[str, object]],
    factor: str,
    factor_values: tuple[int, ...],
    other_factor: str,
    expected_other_values: tuple[int, ...],
) -> list[dict[str, object]]:
    output = []
    for factor_value in factor_values:
        selected = [row for row in rows if int(row[factor]) == factor_value]
        observed_other = {int(row[other_factor]) for row in selected}
        if observed_other != set(expected_other_values):
            raise RuntimeError(f"incomplete matrix for {factor}={factor_value}")
        scores = [float(row["validation_macro_f1"]) for row in selected]
        output.append(
            {
                factor: factor_value,
                "role": (
                    "development"
                    if factor == "model_seed"
                    and factor_value == DEVELOPMENT_MODEL_SEED
                    else "confirmatory"
                    if factor == "model_seed"
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


def marginal_variability(
    rows: list[dict[str, object]], factor: str, apply_thresholds: bool
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
    if apply_thresholds:
        result.update(
            {
                "sample_std_threshold": prior.STABILITY_SD_THRESHOLD,
                "range_threshold": prior.STABILITY_RANGE_THRESHOLD,
                "sample_std_pass": sample_sd <= prior.STABILITY_SD_THRESHOLD,
                "range_pass": value_range <= prior.STABILITY_RANGE_THRESHOLD,
                "stable": sample_sd <= prior.STABILITY_SD_THRESHOLD
                and value_range <= prior.STABILITY_RANGE_THRESHOLD,
            }
        )
    return result


def confirmatory_summary(
    model_summary: list[dict[str, object]],
) -> dict[str, object]:
    by_seed = {int(row["model_seed"]): row for row in model_summary}
    development = float(
        by_seed[DEVELOPMENT_MODEL_SEED]["validation_mean_macro_f1"]
    )
    confirmatory = [
        float(by_seed[seed]["validation_mean_macro_f1"])
        for seed in CONFIRMATORY_MODEL_SEEDS
    ]
    confirmatory_mean = statistics.mean(confirmatory)
    return {
        "development_model_seed": DEVELOPMENT_MODEL_SEED,
        "development_marginal_mean_macro_f1": development,
        "confirmatory_model_seeds": list(CONFIRMATORY_MODEL_SEEDS),
        "confirmatory_mean_of_model_seed_marginals_macro_f1": confirmatory_mean,
        "confirmatory_model_seed_marginal_sample_std_macro_f1": base.sample_std(
            confirmatory
        ),
        "confirmatory_min_model_seed_marginal_macro_f1": min(confirmatory),
        "confirmatory_max_model_seed_marginal_macro_f1": max(confirmatory),
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
    expected_cells = len(base.SPLIT_SEEDS) * len(MODEL_SEEDS)
    if len(scores) != expected_cells:
        raise RuntimeError("factor decomposition requires the full 3x5 matrix")
    values = list(scores.values())
    grand = statistics.mean(values)
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
    ss_total = sum((value - grand) ** 2 for value in values)
    ss_split = len(MODEL_SEEDS) * sum(
        (value - grand) ** 2 for value in split_means.values()
    )
    ss_model = len(base.SPLIT_SEEDS) * sum(
        (value - grand) ** 2 for value in model_means.values()
    )
    ss_interaction = max(ss_total - ss_split - ss_model, 0.0)
    components = (
        ("split_seed", len(base.SPLIT_SEEDS) - 1, ss_split),
        ("model_seed", len(MODEL_SEEDS) - 1, ss_model),
        (
            "split_by_model_seed_interaction",
            (len(base.SPLIT_SEEDS) - 1) * (len(MODEL_SEEDS) - 1),
            ss_interaction,
        ),
    )
    return {
        "method": (
            "balanced 3x5 two-way descriptive decomposition without replicated "
            "cells; interaction is residual and no inferential test is performed"
        ),
        "balanced_grand_mean_macro_f1": grand,
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


def validation_device_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_device_seed: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        model_seed = int(row["model_seed"])
        group_rows = base.read_csv(Path(str(row["run_dir"])) / "group_metrics.csv")
        for group_row in group_rows:
            if group_row["split"] != "validation" or group_row["grouping"] != "device":
                continue
            by_device_seed.setdefault((group_row["group_id"], model_seed), []).append(
                float(group_row["macro_f1"])
            )
    output = []
    for device_id in sorted({key[0] for key in by_device_seed}):
        model_marginals = []
        for model_seed in MODEL_SEEDS:
            values = by_device_seed.get((device_id, model_seed), [])
            if len(values) != len(base.SPLIT_SEEDS):
                raise RuntimeError(
                    f"incomplete device data for {device_id}, seed {model_seed}"
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
                <= prior.STABILITY_SD_THRESHOLD
                and value_range <= prior.STABILITY_RANGE_THRESHOLD,
            }
        )
    return output


def checkpoint_manifest(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        checkpoint = Path(str(row["run_dir"])) / "final_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing final checkpoint: {checkpoint}")
        output.append(
            {
                "model_seed": int(row["model_seed"]),
                "split_seed": int(row["split_seed"]),
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": base.file_sha256(checkpoint),
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
    source_summary: dict[str, object],
    rows: list[dict[str, object]],
    model_summary: list[dict[str, object]],
    split_summary: list[dict[str, object]],
    model_variability: dict[str, object],
    split_variability: dict[str, object],
    confirmation: dict[str, object],
    deltas: list[dict[str, object]],
    decomposition: dict[str, object],
    devices: list[dict[str, object]],
    status: str,
) -> None:
    scores = {
        (int(row["split_seed"]), int(row["model_seed"])): row[
            "validation_macro_f1"
        ]
        for row in rows
    }
    matrix_rows = [
        [
            split_seed,
            *[fmt(scores[(split_seed, model_seed)]) for model_seed in MODEL_SEEDS],
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
    factor_rows = [
        [
            row["source"],
            row["degrees_of_freedom"],
            f"{float(row['sum_of_squares']):.6f}",
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
    initial = source_summary["model_seed_marginal_stability"]
    stable = bool(model_variability["stable"])
    lines = [
        "# HHAR FedAvg Five-Model-Seed Expansion V1",
        "",
        "## Purpose",
        "",
        "The pre-registered 3x3 pass exceeded its model-seed SD threshold. This "
        "conditional expansion adds model seeds `20260618` and `20260619` without "
        "changing the frozen FedAvg configuration or reading test performance.",
        "",
        "## Frozen Protocol",
        "",
        f"- Split seeds: `{list(base.SPLIT_SEEDS)}`.",
        f"- Model seeds: `{list(MODEL_SEEDS)}`; full `3 x 5` crossed design.",
        "- BatchNorm, SGD momentum `0.9`, constant learning rate `0.01`, 1 local "
        "epoch, 50 rounds, full participation.",
        "- Primary metric: validation global Macro-F1 only.",
        f"- Stability rule: model-seed marginal sample SD <= "
        f"`{prior.STABILITY_SD_THRESHOLD}` and range <= "
        f"`{prior.STABILITY_RANGE_THRESHOLD}`.",
        "- No further seed expansion or FedAvg retuning is allowed after this pass.",
        "",
        "## Validation Matrix",
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
        f"- Initial 3-seed marginal SD/range: "
        f"`{fmt(initial['marginal_sample_std_macro_f1'])}` / "
        f"`{fmt(initial['marginal_range_macro_f1'])}`.",
        f"- Final 5-seed balanced mean: "
        f"`{fmt(model_variability['balanced_grand_mean_macro_f1'])}`.",
        f"- Final model-seed marginal SD/range: "
        f"`{fmt(model_variability['marginal_sample_std_macro_f1'])}` / "
        f"`{fmt(model_variability['marginal_range_macro_f1'])}`.",
        f"- Pre-registered stability pass: `{stable}`.",
        "",
        "Confirmatory estimate, excluding the development seed:",
        "",
        f"- Four-seed confirmatory mean: "
        f"`{fmt(confirmation['confirmatory_mean_of_model_seed_marginals_macro_f1'])}`.",
        f"- Confirmatory model-seed SD: "
        f"`{fmt(confirmation['confirmatory_model_seed_marginal_sample_std_macro_f1'])}`.",
        f"- Confirmatory minus development: "
        f"`{fmt(confirmation['confirmatory_minus_development_macro_f1'])}`.",
        "",
        "## Paired Differences",
        "",
        markdown_table(
            [
                "Model seed",
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
        f"Split marginal SD/range: "
        f"`{fmt(split_variability['marginal_sample_std_macro_f1'])}` / "
        f"`{fmt(split_variability['marginal_range_macro_f1'])}`. Model and split "
        "uncertainty remain separate.",
        "",
        "## Descriptive Factor Decomposition",
        "",
        markdown_table(["Source", "df", "SS", "Share of total SS"], factor_rows),
        "",
        "The decomposition is descriptive; one deterministic run exists per cell, "
        "so interaction is residual and no p-values are reported.",
        "",
        "## Validation By Device",
        "",
        markdown_table(
            ["Device", "Val mean +/- model-seed SD", "Range", "Descriptive stable"],
            device_rows,
        ),
        "",
        "Device results are descriptive and do not gate the global decision.",
        "",
        "## Decision",
        "",
        f"Status: `{status}`.",
        "",
        (
            "The frozen baseline now satisfies the registered model-seed stability "
            "rule."
            if stable
            else "The frozen baseline remains model-seed-sensitive after five-seed "
            "characterization. This is a property to report, not a reason to retune."
        ),
        "The observed model-seed range remains material, so all headline HHAR "
        "comparisons must retain the five-seed distribution.",
        "",
        "The next step is a separately pre-registered, evaluation-only locked test "
        "over all 15 saved final checkpoints. No checkpoint or seed may be selected.",
        "",
        "No pooled 15-cell SD is reported because it would conflate split and "
        "model-seed uncertainty.",
        "",
        "## Audit",
        "",
        "All 15 cells match the frozen split identities, runtime, configuration, "
        "checkpoint schedule, and communication budget. Metrics contain only "
        "`train` and `validation`; test performance remains locked.",
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
        source_summary["frozen_primary_reference"]["total_communication_bytes"]
    )
    rows = []
    for source_row in source_rows:
        rows.append(
            prior.validate_run(
                args,
                Path(str(source_row["run_dir"])),
                int(source_row["model_seed"]),
                int(source_row["split_seed"]),
                split_identity,
                expected_communication,
            )
        )
    print(f"validated {len(rows)} reused 3x3 cells", flush=True)

    ordinal = 0
    total_new = len(ADDITIONAL_MODEL_SEEDS) * len(base.SPLIT_SEEDS)
    for model_seed in ADDITIONAL_MODEL_SEEDS:
        for split_seed in base.SPLIT_SEEDS:
            ordinal += 1
            rows.append(
                prior.run_one(
                    args,
                    model_seed,
                    split_seed,
                    split_identity,
                    expected_communication,
                    f"expansion {ordinal}/{total_new}",
                )
            )
    rows.sort(key=lambda row: (int(row["model_seed"]), int(row["split_seed"])))
    cells = {(int(row["model_seed"]), int(row["split_seed"])) for row in rows}
    expected_cells = {
        (model_seed, split_seed)
        for model_seed in MODEL_SEEDS
        for split_seed in base.SPLIT_SEEDS
    }
    if cells != expected_cells:
        raise RuntimeError("five-model-seed matrix is incomplete")
    if any(bool(row["test_metrics_generated"]) for row in rows):
        raise RuntimeError("test metric leakage detected")
    resolved_devices = {str(row["resolved_device"]) for row in rows}
    if len(resolved_devices) != 1:
        raise RuntimeError(f"resolved device changed across cells: {resolved_devices}")

    model_summary = aggregate_by_factor(
        rows, "model_seed", MODEL_SEEDS, "split_seed", base.SPLIT_SEEDS
    )
    split_summary = aggregate_by_factor(
        rows, "split_seed", base.SPLIT_SEEDS, "model_seed", MODEL_SEEDS
    )
    model_variability = marginal_variability(model_summary, "model_seed", True)
    split_variability = marginal_variability(split_summary, "split_seed", False)
    confirmation = confirmatory_summary(model_summary)
    deltas = paired_deltas(rows)
    decomposition = two_way_decomposition(rows)
    devices = validation_device_summary(rows)
    checkpoints = checkpoint_manifest(rows)
    stable = bool(model_variability["stable"])
    status = (
        "five_model_seed_characterization_complete_stable_ready_for_locked_test"
        if stable
        else "five_model_seed_characterization_complete_seed_sensitive_ready_for_locked_test"
    )

    base.write_csv(args.output_root / "run_results.csv", rows)
    base.write_csv(args.output_root / "model_seed_summary.csv", model_summary)
    base.write_csv(args.output_root / "split_seed_summary.csv", split_summary)
    base.write_csv(args.output_root / "paired_deltas.csv", deltas)
    base.write_csv(args.output_root / "validation_device_summary.csv", devices)
    base.write_csv(args.output_root / "checkpoint_manifest.csv", checkpoints)
    summary = {
        "status": status,
        "pre_registration": registered,
        "test_access_audit": {
            "evaluation_splits": list(base.EVALUATION_SPLITS),
            "test_metrics_generated": False,
            "all_fifteen_cells_validated": True,
        },
        "frozen_primary_reference": {
            **prior.CONFIG,
            "rounds": prior.TARGET_ROUNDS,
            "local_epochs": prior.LOCAL_EPOCHS,
            "total_communication_bytes": expected_communication,
            "resolved_device": next(iter(resolved_devices)),
        },
        "run_results": rows,
        "model_seed_summary": model_summary,
        "split_seed_summary": split_summary,
        "model_seed_marginal_variability": model_variability,
        "split_seed_marginal_variability": split_variability,
        "confirmatory_summary": confirmation,
        "paired_model_seed_deltas": deltas,
        "two_way_decomposition": decomposition,
        "validation_device_model_seed_summary": devices,
        "checkpoint_manifest": checkpoints,
        "next_step": (
            "separately pre-register evaluation-only locked test over all 15 "
            "saved checkpoints"
        ),
    }
    base.write_json(args.output_root / "summary.json", summary)
    base.write_json(args.published_summary_path, summary)
    write_report(
        args.report_path,
        source_summary,
        rows,
        model_summary,
        split_summary,
        model_variability,
        split_variability,
        confirmation,
        deltas,
        decomposition,
        devices,
        status,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
