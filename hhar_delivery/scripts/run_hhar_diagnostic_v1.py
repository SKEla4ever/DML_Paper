#!/usr/bin/env python3
"""Run setup-health diagnostics for the frozen HHAR V1 pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load script at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


diagnostic = load_script(
    "kuhar_diagnostic_shared",
    REPO_ROOT / "kuhar_delivery" / "scripts" / "run_kuhar_diagnostic_v1.py",
)
hhar = load_script(
    "hhar_fedavg_loader",
    REPO_ROOT
    / "algorithms"
    / "1d_cnn_fedavg"
    / "train_hhar_1d_cnn_fedavg.py",
)
base = diagnostic.base


DEFAULT_OUTPUT_DIR = Path("outputs/hhar_diagnostic_v1")
DEFAULT_REPORT_PATH = Path("hhar_delivery/reports/hhar/hhar_diagnostic_v1.md")
RANDOM_LABEL_THRESHOLD = 0.30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=hhar.DEFAULT_MANIFEST_DIR)
    parser.add_argument("--archive", type=Path, default=hhar.DEFAULT_ARCHIVE)
    parser.add_argument("--cache-dir", type=Path, default=hhar.DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument(
        "--norm", choices=("batchnorm", "groupnorm", "none"), default="batchnorm"
    )
    parser.add_argument("--groupnorm-groups", type=int, default=8)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--centralized-epochs", type=int, default=15)
    parser.add_argument("--random-label-epochs", type=int, default=5)
    parser.add_argument("--tiny-overfit-epochs", type=int, default=150)
    parser.add_argument("--tiny-overfit-lr", type=float, default=0.003)
    parser.add_argument("--tiny-overfit-classes", type=int, default=4)
    parser.add_argument("--tiny-overfit-windows-per-class", type=int, default=16)
    parser.add_argument("--fedavg-rounds", type=int, default=10)
    parser.add_argument("--fedavg-local-epochs", type=int, default=1)
    parser.add_argument("--fedavg-eval-every", type=int, default=5)
    parser.add_argument("--iid-clients", type=int, default=69)
    parser.add_argument(
        "--experiment-seed-isolation",
        action="store_true",
        help="Reset model RNG before each diagnostic experiment.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[hhar_diagnostic_v1] {message}", flush=True)


def reset_experiment_seed(args: argparse.Namespace, device: object) -> None:
    if not args.experiment_seed_isolation:
        return
    base.torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        base.torch.cuda.manual_seed_all(args.seed)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def feature_shift_rows(
    windows: np.ndarray,
    devices: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    global_mean = windows[train_indices].mean(axis=(0, 2))
    rows = []
    distances = []
    for device_id in sorted(set(devices[train_indices].tolist())):
        indices = train_indices[devices[train_indices] == device_id]
        device_windows = windows[indices]
        means = device_windows.mean(axis=(0, 2))
        stds = device_windows.std(axis=(0, 2))
        distance = float(np.linalg.norm(means - global_mean))
        distances.append(distance)
        rows.append(
            {
                "device_id": device_id,
                "train_windows": len(indices),
                "standardized_mean_x": float(means[0]),
                "standardized_mean_y": float(means[1]),
                "standardized_mean_z": float(means[2]),
                "standardized_std_x": float(stds[0]),
                "standardized_std_y": float(stds[1]),
                "standardized_std_z": float(stds[2]),
                "mean_l2_distance_from_global": distance,
            }
        )
    return rows, {
        "min_device_mean_l2_distance": min(distances),
        "max_device_mean_l2_distance": max(distances),
        "device_mean_l2_distance_range": max(distances) - min(distances),
    }


def evaluate_experiment(
    name: str,
    model: object,
    x_tensor: object,
    labels: np.ndarray,
    y_tensor: object,
    clients: np.ndarray,
    split_indices: dict[str, np.ndarray],
    eval_clients: set[str],
    device: object,
    args: argparse.Namespace,
    class_names: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]], dict]:
    return diagnostic.evaluate_model_all_splits(
        experiment=name,
        model=model,
        x_tensor=x_tensor,
        y=labels,
        y_tensor=y_tensor,
        subjects=clients,
        split_indices=split_indices,
        eval_subjects=eval_clients,
        device=device,
        batch_size=args.batch_size,
        num_classes=len(class_names),
        class_names=class_names,
    )


def add_group_metrics(
    output: list[dict[str, object]],
    experiment: str,
    model: object,
    windows: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    devices: np.ndarray,
    split_indices: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    for row in hhar.group_metric_rows(
        model=model,
        windows=windows,
        labels=labels,
        users=users,
        devices=devices,
        split_indices=split_indices,
        args=args,
    ):
        output.append({"experiment": experiment, **row})


def metric(metrics: dict[str, Any], split: str, key: str) -> float:
    return float(metrics[split][key])


def health_gates(
    tiny: dict[str, Any],
    randomized: dict[str, Any],
    centralized: dict[str, Any],
    real_fedavg: dict[str, Any],
    iid_fedavg: dict[str, Any],
    feature_summary: dict[str, float],
    manifest_audit: dict[str, Any],
) -> list[dict[str, object]]:
    tiny_accuracy = metric(tiny, "tiny_train", "accuracy")
    tiny_f1 = metric(tiny, "tiny_train", "macro_f1")
    random_val = metric(randomized, "validation", "macro_f1")
    central_val = metric(centralized, "validation", "macro_f1")
    central_test = metric(centralized, "test", "macro_f1")
    real_val = metric(real_fedavg, "validation", "macro_f1")
    real_test = metric(real_fedavg, "test", "macro_f1")
    iid_val = metric(iid_fedavg, "validation", "macro_f1")
    iid_test = metric(iid_fedavg, "test", "macro_f1")
    split_ok = bool(manifest_audit["realized"]["class_support_all_splits"])
    train_client_ok = (
        manifest_audit["realized"]["clients"]
        == manifest_audit["realized"]["training_clients"]
    )
    return [
        {
            "gate": "manifest_integrity",
            "status": "PASS" if split_ok and train_client_ok else "FAIL",
            "criterion": "Every class has all-split support and every retained client has train data.",
            "observed": {
                "class_support_all_splits": split_ok,
                "all_retained_clients_have_train": train_client_ok,
            },
        },
        {
            "gate": "tiny_overfit",
            "status": "PASS" if tiny_accuracy >= 0.95 and tiny_f1 >= 0.95 else "FAIL",
            "criterion": "Tiny-subset accuracy and Macro-F1 both reach at least 0.95.",
            "observed": {"accuracy": tiny_accuracy, "macro_f1": tiny_f1},
        },
        {
            "gate": "random_label_negative_control",
            "status": "PASS" if random_val <= RANDOM_LABEL_THRESHOLD else "FAIL",
            "criterion": f"Random-label validation Macro-F1 is at most {RANDOM_LABEL_THRESHOLD:.2f}.",
            "observed": {
                "validation_macro_f1": random_val,
                "chance_scale": 1 / len(hhar.ACTIVITIES),
            },
        },
        {
            "gate": "centralized_learnability",
            "status": "PASS"
            if central_val >= 0.50 and central_val - random_val >= 0.20
            else "FAIL",
            "criterion": "Centralized validation Macro-F1 reaches 0.50 and exceeds random-label control by 0.20.",
            "observed": {
                "centralized_validation_macro_f1": central_val,
                "random_label_validation_macro_f1": random_val,
                "margin": central_val - random_val,
            },
        },
        {
            "gate": "iid_vs_real_fedavg",
            "status": "PASS" if iid_val >= real_val - 0.03 else "WARN",
            "criterion": "IID-client FedAvg is not more than 0.03 below real client FedAvg at equal communication.",
            "observed": {
                "iid_validation_macro_f1": iid_val,
                "real_validation_macro_f1": real_val,
                "margin": iid_val - real_val,
            },
        },
        {
            "gate": "validation_test_difficulty_alignment",
            "status": "PASS"
            if max(
                abs(central_test - central_val),
                abs(real_test - real_val),
                abs(iid_test - iid_val),
            )
            <= 0.20
            else "WARN",
            "criterion": "Absolute validation/test Macro-F1 gaps stay within 0.20 across centralized, real-client, and IID-client controls.",
            "observed": {
                "centralized_gap": central_test - central_val,
                "real_fedavg_gap": real_test - real_val,
                "iid_fedavg_gap": iid_test - iid_val,
            },
        },
        {
            "gate": "device_feature_shift_present",
            "status": "PASS"
            if feature_summary["max_device_mean_l2_distance"] >= 0.10
            else "WARN",
            "criterion": "At least one device train mean is visibly shifted after global train standardization (L2 >= 0.10).",
            "observed": feature_summary,
        },
    ]


def write_report(
    path: Path,
    args: argparse.Namespace,
    manifest_summary: dict[str, object],
    gates: list[dict[str, object]],
    overall_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    distribution_summary: dict[str, object],
    feature_summary: dict[str, float],
    elapsed_seconds: float,
) -> None:
    indexed = {(row["experiment"], row["split"]): row for row in overall_rows}
    comparison = []
    for experiment, label in (
        ("centralized_oracle", "centralized"),
        ("real_client_fedavg", "user-device FedAvg"),
        ("iid_client_fedavg", "IID-client FedAvg"),
        ("random_label_centralized", "random-label centralized"),
    ):
        val = indexed[(experiment, "validation")]
        test = indexed[(experiment, "test")]
        comparison.append(
            [
                label,
                f"{float(val['macro_f1']):.4f}",
                f"{float(test['macro_f1']):.4f}",
                f"{int(val['total_communication_bytes'] or 0):,}",
            ]
        )
    gate_table = [
        [gate["gate"], gate["status"], json.dumps(gate["observed"], sort_keys=True)]
        for gate in gates
    ]
    fedavg_devices = [
        row
        for row in group_rows
        if row["experiment"] == "real_client_fedavg"
        and row["split"] == "test"
        and row["grouping"] == "device"
    ]
    device_table = [
        [
            row["group_id"],
            row["windows"],
            row["supported_classes"],
            f"{float(row['macro_f1']):.4f}",
        ]
        for row in fedavg_devices
    ]
    all_pass = all(gate["status"] == "PASS" for gate in gates)
    lines = [
        "# HHAR Diagnostic V1",
        "",
        "## Purpose",
        "",
        "This diagnostic checks the frozen HHAR V1 raw parser, synchronized split, "
        "resampling, neural model, controls, and communication accounting before "
        "algorithm tuning. It is not the final tuned FedAvg result.",
        "",
        "## Protocol",
        "",
        f"- Seed: `{args.seed}`; BatchNorm/Adam learning rate `{args.lr}`.",
        f"- Centralized epochs: `{args.centralized_epochs}`; random-label epochs: "
        f"`{args.random_label_epochs}`.",
        f"- Real and IID FedAvg: `{args.fedavg_rounds}` rounds, "
        f"`{args.fedavg_local_epochs}` local epoch, `{args.iid_clients}` clients.",
        f"- Windows: `{manifest_summary['windows']}`; split windows: "
        f"`{manifest_summary['split_windows']}`.",
        f"- Elapsed wall time: `{elapsed_seconds:.1f}` seconds.",
        "",
        "## Health Gates",
        "",
        diagnostic.markdown_table(["Gate", "Status", "Observed"], gate_table),
        "",
        "## Main Metrics",
        "",
        diagnostic.markdown_table(
            ["Protocol", "Val Macro-F1", "Test Macro-F1", "Communication bytes"],
            comparison,
        ),
        "",
        "## FedAvg Test By Device",
        "",
        diagnostic.markdown_table(
            ["Device", "Windows", "Supported classes", "Macro-F1"], device_table
        ),
        "",
        "## Diagnosis",
        "",
        "All health gates pass."
        if all_pass
        else "At least one health gate is not PASS; treat the diagnostic as actionable evidence rather than a clean bill of health.",
        "",
        "HHAR retains explicit device feature shift after train-only global channel "
        "standardization: the largest device mean L2 distance is "
        f"`{feature_summary['max_device_mean_l2_distance']:.4f}`. The client label "
        "distribution is also non-IID under the synchronized execution split: mean "
        "JS divergence from global train is "
        f"`{float(distribution_summary['mean_js_divergence_from_global_train']):.4f}`.",
        "",
        "Pair-level metrics are descriptive because 69 retained user-device clients "
        "come from only nine physical users. Global Macro-F1 and per-device behavior "
        "are the primary HHAR outcomes; this dataset is not used for independent-user "
        "fairness claims.",
        "",
        "Validation is systematically harder than test across the centralized, "
        "real-client FedAvg, and IID-client controls. This is consistent with high "
        "execution-group variance in a nine-user dataset. Before selecting "
        "hyperparameters, run a pre-registered split-seed sensitivity pass and report "
        "aggregate behavior rather than relying on this single split alone.",
        "",
        "## Artifacts",
        "",
        f"- Overall metrics: `{args.output_dir / 'overall_metrics.csv'}`",
        f"- Device/user metrics: `{args.output_dir / 'group_metrics.csv'}`",
        f"- Device feature statistics: `{args.output_dir / 'device_feature_shift.csv'}`",
        f"- Health gates: `{args.output_dir / 'health_gates.json'}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    start = time.time()
    final_summary_path = args.output_dir / "diagnostic_summary.json"
    if final_summary_path.exists() and not args.force:
        print(f"diagnostic already exists: {final_summary_path}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log("loading frozen HHAR manifest")
    rows, eval_clients, test_supported_clients = hhar.read_window_manifest(
        args.manifest_dir
    )
    manifest_summary = hhar.manifest_summary(
        rows, eval_clients, test_supported_clients
    )
    write_json(args.output_dir / "manifest_sanity.json", manifest_summary)
    if args.dry_run:
        print(json.dumps(manifest_summary, indent=2, sort_keys=True))
        return

    base.require_torch()
    device = base.choose_device(args.device)
    base.torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    labels, clients, users, devices, split_indices = hhar.build_label_arrays(rows)
    log(f"loading resampled windows on device={device}")
    windows = hhar.load_or_build_windows(
        rows, args.archive, hhar.cache_path(args.cache_dir), args.rebuild_cache
    )
    windows, standardization = base.standardize_windows(
        windows, split_indices["train"]
    )
    write_json(args.output_dir / "channel_standardization.json", standardization)

    x_tensor = base.torch.from_numpy(windows)
    y_tensor = base.torch.from_numpy(labels)
    class_names = {index: activity for index, activity in enumerate(hhar.ACTIVITIES)}
    log("computing client-label and device-feature diagnostics")
    distribution_summary, client_distribution, class_distribution = (
        diagnostic.distribution_diagnostics(
            labels,
            clients,
            split_indices,
            len(class_names),
            class_names,
        )
    )
    feature_rows, feature_summary = feature_shift_rows(
        windows, devices, split_indices["train"]
    )
    write_json(args.output_dir / "distribution_summary.json", distribution_summary)
    write_csv(args.output_dir / "client_label_distribution.csv", client_distribution)
    write_csv(args.output_dir / "class_distribution.csv", class_distribution)
    write_csv(args.output_dir / "device_feature_shift.csv", feature_rows)

    overall_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    per_client_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    confusion: dict[str, Any] = {}
    histories: dict[str, Any] = {}

    log("running tiny-overfit positive control")
    reset_experiment_seed(args, device)
    tiny_indices = diagnostic.select_tiny_indices(
        labels,
        split_indices["train"],
        args.seed,
        args.tiny_overfit_classes,
        args.tiny_overfit_windows_per_class,
    )
    tiny_model, histories["tiny_overfit"] = diagnostic.train_tiny_overfit(
        x_tensor,
        labels,
        tiny_indices,
        args,
        device,
        len(class_names),
    )
    tiny_metric, tiny_predictions = diagnostic.evaluate_model_on_indices(
        tiny_model,
        x_tensor,
        labels,
        y_tensor,
        clients,
        tiny_indices,
        set(clients[tiny_indices].astype(str).tolist()),
        device,
        args.batch_size,
        len(class_names),
    )
    tiny_metrics = {"tiny_train": tiny_metric}
    diagnostic.append_overall_rows(
        overall_rows, "tiny_overfit", tiny_metrics, "tiny overfit"
    )
    confusion["tiny_overfit"] = {
        "tiny_train": diagnostic.confusion_matrix(
            labels[tiny_indices], tiny_predictions, len(class_names)
        ).tolist()
    }

    log("running centralized oracle")
    reset_experiment_seed(args, device)
    central_model, histories["centralized_oracle"] = diagnostic.train_centralized(
        "centralized_oracle",
        x_tensor,
        labels,
        labels,
        split_indices,
        args,
        device,
        len(class_names),
        args.centralized_epochs,
    )
    central_metrics, rows_pc, rows_pu, matrices = evaluate_experiment(
        "centralized_oracle",
        central_model,
        x_tensor,
        labels,
        y_tensor,
        clients,
        split_indices,
        eval_clients,
        device,
        args,
        class_names,
    )
    diagnostic.append_overall_rows(
        overall_rows, "centralized_oracle", central_metrics, "centralized"
    )
    per_class_rows.extend(rows_pc)
    per_client_rows.extend(rows_pu)
    confusion["centralized_oracle"] = matrices
    add_group_metrics(
        group_rows,
        "centralized_oracle",
        central_model,
        windows,
        labels,
        users,
        devices,
        split_indices,
        args,
    )

    log("running random-label negative control")
    reset_experiment_seed(args, device)
    randomized_labels = diagnostic.make_randomized_train_labels(
        labels, split_indices["train"], args.seed
    )
    random_model, histories["random_label_centralized"] = (
        diagnostic.train_centralized(
            "random_label_centralized",
            x_tensor,
            labels,
            randomized_labels,
            split_indices,
            args,
            device,
            len(class_names),
            args.random_label_epochs,
        )
    )
    random_metrics, rows_pc, rows_pu, matrices = evaluate_experiment(
        "random_label_centralized",
        random_model,
        x_tensor,
        labels,
        y_tensor,
        clients,
        split_indices,
        eval_clients,
        device,
        args,
        class_names,
    )
    diagnostic.append_overall_rows(
        overall_rows,
        "random_label_centralized",
        random_metrics,
        "random-label centralized",
    )
    per_class_rows.extend(rows_pc)
    per_client_rows.extend(rows_pu)
    confusion["random_label_centralized"] = matrices

    log("running real user-device FedAvg")
    reset_experiment_seed(args, device)
    real_history, real_model = diagnostic.run_fedavg_diagnostic(
        "real_client_fedavg",
        windows,
        labels,
        clients,
        split_indices,
        eval_clients,
        args,
    )
    histories["real_client_fedavg"] = real_history
    real_metrics, rows_pc, rows_pu, matrices = evaluate_experiment(
        "real_client_fedavg",
        real_model,
        x_tensor,
        labels,
        y_tensor,
        clients,
        split_indices,
        eval_clients,
        device,
        args,
        class_names,
    )
    real_communication = real_history[-1]["communication"]["total_bytes"]
    diagnostic.append_overall_rows(
        overall_rows,
        "real_client_fedavg",
        real_metrics,
        "user-device FedAvg",
        real_communication,
    )
    per_class_rows.extend(rows_pc)
    per_client_rows.extend(rows_pu)
    confusion["real_client_fedavg"] = matrices
    add_group_metrics(
        group_rows,
        "real_client_fedavg",
        real_model,
        windows,
        labels,
        users,
        devices,
        split_indices,
        args,
    )

    log("running equal-client-count IID FedAvg")
    reset_experiment_seed(args, device)
    iid_clients = diagnostic.make_iid_train_subjects(
        clients,
        labels,
        split_indices["train"],
        args.iid_clients,
        args.seed,
    )
    actual_iid_clients = diagnostic.count_train_clients(
        iid_clients, split_indices["train"]
    )
    if actual_iid_clients != args.iid_clients:
        raise RuntimeError(
            f"expected {args.iid_clients} IID clients, got {actual_iid_clients}"
        )
    iid_history, iid_model = diagnostic.run_fedavg_diagnostic(
        "iid_client_fedavg",
        windows,
        labels,
        iid_clients,
        split_indices,
        eval_clients,
        args,
    )
    histories["iid_client_fedavg"] = iid_history
    iid_metrics, rows_pc, rows_pu, matrices = evaluate_experiment(
        "iid_client_fedavg",
        iid_model,
        x_tensor,
        labels,
        y_tensor,
        clients,
        split_indices,
        eval_clients,
        device,
        args,
        class_names,
    )
    iid_communication = iid_history[-1]["communication"]["total_bytes"]
    if iid_communication != real_communication:
        raise RuntimeError(
            "real and IID FedAvg communication differ despite equal client counts"
        )
    diagnostic.append_overall_rows(
        overall_rows,
        "iid_client_fedavg",
        iid_metrics,
        "IID-client FedAvg",
        iid_communication,
    )
    per_class_rows.extend(rows_pc)
    per_client_rows.extend(rows_pu)
    confusion["iid_client_fedavg"] = matrices
    add_group_metrics(
        group_rows,
        "iid_client_fedavg",
        iid_model,
        windows,
        labels,
        users,
        devices,
        split_indices,
        args,
    )

    manifest_audit = json.loads(
        (args.manifest_dir / "hhar_split_audit_v1.json").read_text()
    )
    gates = health_gates(
        tiny_metrics,
        random_metrics,
        central_metrics,
        real_metrics,
        iid_metrics,
        feature_summary,
        manifest_audit,
    )
    write_csv(args.output_dir / "overall_metrics.csv", overall_rows)
    write_csv(args.output_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(args.output_dir / "per_client_metrics.csv", per_client_rows)
    write_csv(args.output_dir / "group_metrics.csv", group_rows)
    write_json(args.output_dir / "confusion_matrices.json", confusion)
    write_json(args.output_dir / "training_histories.json", histories)
    write_json(args.output_dir / "health_gates.json", gates)

    elapsed = time.time() - start
    config = {
        "dataset": "HHAR",
        "manifest_version": "V1",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "norm": args.norm,
        "centralized_epochs": args.centralized_epochs,
        "random_label_epochs": args.random_label_epochs,
        "tiny_overfit_epochs": args.tiny_overfit_epochs,
        "fedavg_rounds": args.fedavg_rounds,
        "fedavg_local_epochs": args.fedavg_local_epochs,
        "iid_clients": args.iid_clients,
        "experiment_seed_isolation": args.experiment_seed_isolation,
        "resolved_device": str(device),
        "input_shape": [3, hhar.WINDOW_SAMPLES],
        "interpretation": "setup diagnostic, not tuned baseline",
    }
    summary = {
        "run_config": config,
        "manifest_summary": manifest_summary,
        "distribution_summary": distribution_summary,
        "feature_shift_summary": feature_summary,
        "health_gates": gates,
        "elapsed_seconds": elapsed,
    }
    write_json(final_summary_path, summary)
    write_report(
        args.report_path,
        args,
        manifest_summary,
        gates,
        overall_rows,
        group_rows,
        distribution_summary,
        feature_summary,
        elapsed,
    )
    log("complete")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
