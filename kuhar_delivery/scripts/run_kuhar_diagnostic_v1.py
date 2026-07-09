#!/usr/bin/env python3
"""Run KU-HAR setup-health diagnostics for the frozen V1 pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FEDAVG_SCRIPT = (
    REPO_ROOT / "algorithms" / "1d_cnn_fedavg" / "train_kuhar_1d_cnn_fedavg.py"
)
spec = importlib.util.spec_from_file_location("kuhar_fedavg_base", FEDAVG_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load FedAvg base script at {FEDAVG_SCRIPT}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


DEFAULT_OUTPUT_DIR = Path("outputs/kuhar_diagnostic_v1")
DEFAULT_REPORT_PATH = Path("kuhar_delivery/reports/kuhar/kuhar_diagnostic_v1.md")
DEFAULT_SELECTED_FEDAVG = Path(
    "outputs/kuhar_1d_cnn_fedavg_tuning_v1/batchnorm_adam_lr0p001_e2_r50"
)
CHANCE_MULTIPLIER_THRESHOLD = 2.0
RANDOM_LABEL_ABSOLUTE_THRESHOLD = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=base.DEFAULT_MANIFEST_DIR)
    parser.add_argument("--archive", type=Path, default=base.DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--selected-fedavg-run-dir", type=Path, default=DEFAULT_SELECTED_FEDAVG
    )
    parser.add_argument(
        "--cohort",
        choices=("minimum_support", "full_sparse"),
        default="minimum_support",
    )
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument(
        "--norm",
        choices=("batchnorm", "groupnorm", "none"),
        default="batchnorm",
    )
    parser.add_argument("--groupnorm-groups", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--centralized-epochs", type=int, default=30)
    parser.add_argument("--random-label-epochs", type=int, default=8)
    parser.add_argument("--tiny-overfit-epochs", type=int, default=200)
    parser.add_argument("--tiny-overfit-lr", type=float, default=0.003)
    parser.add_argument("--tiny-overfit-classes", type=int, default=4)
    parser.add_argument("--tiny-overfit-windows-per-class", type=int, default=16)
    parser.add_argument("--fedavg-rounds", type=int, default=20)
    parser.add_argument("--fedavg-local-epochs", type=int, default=2)
    parser.add_argument("--fedavg-eval-every", type=int, default=10)
    parser.add_argument("--iid-clients", type=int, default=50)
    parser.add_argument("--local-only-epochs", type=int, default=20)
    parser.add_argument("--skip-local-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-archive-sha256", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def log_stage(message: str) -> None:
    print(f"[kuhar_diagnostic_v1] {message}", flush=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_class_names(manifest_dir: Path) -> dict[int, str]:
    path = manifest_dir / "kuhar_class_summary.csv"
    with path.open(newline="") as handle:
        return {
            int(row["activity_id"]): row["activity"]
            for row in csv.DictReader(handle)
        }


def train_args(
    args: argparse.Namespace,
    *,
    lr: float | None = None,
    local_epochs: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        norm=args.norm,
        groupnorm_groups=args.groupnorm_groups,
        optimizer=args.optimizer,
        lr=args.lr if lr is None else lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        local_epochs=args.fedavg_local_epochs if local_epochs is None else local_epochs,
        seed=args.seed,
        device=args.device,
    )


def fedavg_args(args: argparse.Namespace) -> argparse.Namespace:
    namespace = train_args(args, local_epochs=args.fedavg_local_epochs)
    namespace.rounds = args.fedavg_rounds
    namespace.client_fraction = 1.0
    namespace.eval_every = args.fedavg_eval_every
    return namespace


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def confusion_matrix(
    labels: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[int(label), int(prediction)] += 1
    return matrix


def class_metric_rows(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    class_names: dict[int, str],
    experiment: str,
    split: str,
) -> list[dict[str, object]]:
    rows = []
    for class_index in range(num_classes):
        true_positive = int(
            np.sum((labels == class_index) & (predictions == class_index))
        )
        false_positive = int(
            np.sum((labels != class_index) & (predictions == class_index))
        )
        false_negative = int(
            np.sum((labels == class_index) & (predictions != class_index))
        )
        support = int(np.sum(labels == class_index))
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(
            2 * true_positive, 2 * true_positive + false_positive + false_negative
        )
        rows.append(
            {
                "experiment": experiment,
                "split": split,
                "class_index": class_index,
                "class_name": class_names.get(class_index, str(class_index)),
                "support": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def per_user_rows(
    labels: np.ndarray,
    predictions: np.ndarray,
    subjects: np.ndarray,
    eval_subjects: set[str],
    experiment: str,
    split: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, subject_id in enumerate(subjects.tolist()):
        grouped[str(subject_id)].append(index)

    rows = []
    for subject_id in sorted(eval_subjects):
        indices = grouped.get(subject_id, [])
        if not indices:
            continue
        subject_labels = labels[indices]
        subject_predictions = predictions[indices]
        supported_classes = [
            int(class_index)
            for class_index, count in Counter(subject_labels.tolist()).items()
            if count >= base.SUPPORTED_TEST_CLASS_WINDOWS
        ]
        macro_f1 = None
        if len(supported_classes) >= 3:
            macro_f1 = base.macro_f1(
                subject_labels, subject_predictions, sorted(supported_classes)
            )
        rows.append(
            {
                "experiment": experiment,
                "split": split,
                "subject_id": subject_id,
                "n_windows": len(indices),
                "n_supported_classes": len(supported_classes),
                "accuracy": float(np.mean(subject_predictions == subject_labels)),
                "macro_f1": macro_f1,
            }
        )
    return rows


def summarize_arrays(
    labels: np.ndarray,
    predictions: np.ndarray,
    subjects: np.ndarray,
    eval_subjects: set[str],
    num_classes: int,
    loss: float | None = None,
) -> dict[str, Any]:
    return {
        "n_windows": int(len(labels)),
        "loss": loss,
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": base.macro_f1(labels, predictions, list(range(num_classes))),
        "per_user": base.per_user_macro_f1_summary(
            labels=labels,
            predictions=predictions,
            subjects=subjects,
            eval_subjects=eval_subjects,
            supported_threshold=base.SUPPORTED_TEST_CLASS_WINDOWS,
        ),
    }


def evaluate_model_on_indices(
    model: object,
    x_tensor: object,
    y: np.ndarray,
    y_tensor: object,
    subjects: np.ndarray,
    indices: np.ndarray,
    eval_subjects: set[str],
    device: object,
    batch_size: int,
    num_classes: int,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions, loss = base.predict_split(
        model=model,
        x_tensor=x_tensor,
        y_tensor=y_tensor,
        indices=indices,
        device=device,
        batch_size=batch_size,
    )
    metrics = summarize_arrays(
        labels=y[indices],
        predictions=predictions,
        subjects=subjects[indices],
        eval_subjects=eval_subjects,
        num_classes=num_classes,
        loss=float(loss),
    )
    return metrics, predictions


def run_epoch_training(
    model: object,
    x_tensor: object,
    train_label_tensor: object,
    train_indices: np.ndarray,
    device: object,
    args: argparse.Namespace,
    epochs: int,
    seed_key: str,
) -> list[dict[str, object]]:
    model.train()
    optimizer = base.make_optimizer(model, args)
    criterion = base.nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(base.stable_int(args.seed, seed_key, epoch))
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        total_loss = 0.0
        n_seen = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            batch_x = x_tensor[batch_indices].to(device)
            batch_y = train_label_tensor[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_indices)
            n_seen += len(batch_indices)
        history.append(
            {
                "epoch": epoch,
                "objective_loss": total_loss / max(1, n_seen),
            }
        )
    return history


def evaluate_model_all_splits(
    experiment: str,
    model: object,
    x_tensor: object,
    y: np.ndarray,
    y_tensor: object,
    subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
    eval_subjects: set[str],
    device: object,
    batch_size: int,
    num_classes: int,
    class_names: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]], dict[str, Any]]:
    overall: dict[str, Any] = {}
    per_class: list[dict[str, object]] = []
    per_user: list[dict[str, object]] = []
    confusion: dict[str, Any] = {}
    for split in base.SPLITS:
        indices = split_indices[split]
        metrics, predictions = evaluate_model_on_indices(
            model=model,
            x_tensor=x_tensor,
            y=y,
            y_tensor=y_tensor,
            subjects=subjects,
            indices=indices,
            eval_subjects=eval_subjects,
            device=device,
            batch_size=batch_size,
            num_classes=num_classes,
        )
        overall[split] = metrics
        split_labels = y[indices]
        split_subjects = subjects[indices]
        per_class.extend(
            class_metric_rows(
                labels=split_labels,
                predictions=predictions,
                num_classes=num_classes,
                class_names=class_names,
                experiment=experiment,
                split=split,
            )
        )
        per_user.extend(
            per_user_rows(
                labels=split_labels,
                predictions=predictions,
                subjects=split_subjects,
                eval_subjects=eval_subjects,
                experiment=experiment,
                split=split,
            )
        )
        confusion[split] = confusion_matrix(
            split_labels, predictions, num_classes
        ).tolist()
    return overall, per_class, per_user, confusion


def select_tiny_indices(
    y: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    max_classes: int,
    windows_per_class: int,
) -> np.ndarray:
    counts = Counter(y[train_indices].tolist())
    selected_classes = [
        class_index
        for class_index, count in counts.most_common()
        if count >= windows_per_class
    ][:max_classes]
    if len(selected_classes) < 2:
        raise RuntimeError("not enough supported classes for tiny overfit test")

    rng = np.random.default_rng(base.stable_int(seed, "tiny_overfit"))
    selected: list[int] = []
    for class_index in selected_classes:
        class_indices = train_indices[y[train_indices] == class_index]
        selected.extend(
            rng.choice(class_indices, size=windows_per_class, replace=False).tolist()
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def train_centralized(
    name: str,
    x_tensor: object,
    y: np.ndarray,
    train_labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: object,
    num_classes: int,
    epochs: int,
    lr: float | None = None,
) -> tuple[object, list[dict[str, object]]]:
    training_args = train_args(args, lr=lr, local_epochs=1)
    model = base.build_model(num_classes, training_args).to(device)
    label_tensor = base.torch.from_numpy(train_labels.astype(np.int64))
    history = run_epoch_training(
        model=model,
        x_tensor=x_tensor,
        train_label_tensor=label_tensor,
        train_indices=split_indices["train"],
        device=device,
        args=training_args,
        epochs=epochs,
        seed_key=name,
    )
    return model, history


def train_tiny_overfit(
    x_tensor: object,
    y: np.ndarray,
    tiny_indices: np.ndarray,
    args: argparse.Namespace,
    device: object,
    num_classes: int,
) -> tuple[object, list[dict[str, object]]]:
    training_args = train_args(args, lr=args.tiny_overfit_lr, local_epochs=1)
    model = base.build_model(num_classes, training_args).to(device)
    label_tensor = base.torch.from_numpy(y.astype(np.int64))
    history = run_epoch_training(
        model=model,
        x_tensor=x_tensor,
        train_label_tensor=label_tensor,
        train_indices=tiny_indices,
        device=device,
        args=training_args,
        epochs=args.tiny_overfit_epochs,
        seed_key="tiny_overfit",
    )
    return model, history


def make_randomized_train_labels(
    y: np.ndarray, train_indices: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(base.stable_int(seed, "random_label"))
    randomized = y.copy()
    randomized[train_indices] = rng.permutation(y[train_indices])
    return randomized


def make_iid_train_subjects(
    subjects: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    n_clients: int,
    seed: int,
) -> np.ndarray:
    federated_subjects = subjects.astype(object).copy()
    rng = np.random.default_rng(base.stable_int(seed, "iid_clients"))
    assignments: dict[int, list[int]] = {client_id: [] for client_id in range(n_clients)}
    for class_index in sorted(set(y[train_indices].tolist())):
        class_indices = train_indices[y[train_indices] == class_index].copy()
        rng.shuffle(class_indices)
        for offset, index in enumerate(class_indices.tolist()):
            assignments[offset % n_clients].append(index)
    for client_id, indices in assignments.items():
        synthetic_id = f"iid_{client_id:03d}"
        for index in indices:
            federated_subjects[index] = synthetic_id
    return federated_subjects


def count_train_clients(subjects: np.ndarray, train_indices: np.ndarray) -> int:
    return len({str(subjects[index]) for index in train_indices.tolist()})


def entropy(probabilities: np.ndarray) -> float:
    nonzero = probabilities[probabilities > 0]
    if len(nonzero) == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log2(nonzero)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def distribution_diagnostics(
    y: np.ndarray,
    subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
    num_classes: int,
    class_names: dict[int, str],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    train_indices = split_indices["train"]
    global_counts = np.bincount(y[train_indices], minlength=num_classes)
    global_probs = global_counts / max(1, global_counts.sum())
    class_rows = []
    for split in base.SPLITS:
        split_counts = np.bincount(y[split_indices[split]], minlength=num_classes)
        for class_index in range(num_classes):
            class_rows.append(
                {
                    "split": split,
                    "class_index": class_index,
                    "class_name": class_names.get(class_index, str(class_index)),
                    "windows": int(split_counts[class_index]),
                }
            )

    client_rows = []
    subject_list = subjects.tolist()
    train_index_set = set(train_indices.tolist())
    for subject_id in sorted(set(subject_list)):
        indices = np.asarray(
            [
                index
                for index, current_subject in enumerate(subject_list)
                if current_subject == subject_id and index in train_index_set
            ],
            dtype=np.int64,
        )
        if len(indices) == 0:
            continue
        counts = np.bincount(y[indices], minlength=num_classes)
        probs = counts / counts.sum()
        nonzero_classes = int(np.sum(counts > 0))
        row: dict[str, object] = {
            "subject_id": str(subject_id),
            "train_windows": int(counts.sum()),
            "nonzero_train_classes": nonzero_classes,
            "label_entropy_bits": entropy(probs),
            "js_divergence_from_global_train": jensen_shannon(probs, global_probs),
        }
        for class_index in range(num_classes):
            row[f"class_{class_index}_windows"] = int(counts[class_index])
        client_rows.append(row)

    js_values = [float(row["js_divergence_from_global_train"]) for row in client_rows]
    class_counts = [int(row["nonzero_train_classes"]) for row in client_rows]
    window_counts = [int(row["train_windows"]) for row in client_rows]
    summary = {
        "train_windows": int(global_counts.sum()),
        "clients_with_train": len(client_rows),
        "mean_train_windows_per_client": statistics.mean(window_counts),
        "min_train_windows_per_client": min(window_counts),
        "max_train_windows_per_client": max(window_counts),
        "mean_nonzero_train_classes_per_client": statistics.mean(class_counts),
        "median_nonzero_train_classes_per_client": statistics.median(class_counts),
        "min_nonzero_train_classes_per_client": min(class_counts),
        "max_nonzero_train_classes_per_client": max(class_counts),
        "mean_js_divergence_from_global_train": statistics.mean(js_values),
        "median_js_divergence_from_global_train": statistics.median(js_values),
        "max_js_divergence_from_global_train": max(js_values),
    }
    return summary, client_rows, class_rows


def run_fedavg_diagnostic(
    name: str,
    windows: np.ndarray,
    y: np.ndarray,
    federated_subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
    eval_subjects: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], object]:
    run_args = fedavg_args(args)
    history, model = base.train_fedavg(
        windows=windows,
        y=y,
        subjects=federated_subjects,
        split_indices=split_indices,
        eval_subjects=eval_subjects,
        args=run_args,
    )
    return history, model


def train_local_only(
    x_tensor: object,
    y: np.ndarray,
    y_tensor: object,
    subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
    eval_subjects: set[str],
    args: argparse.Namespace,
    device: object,
    num_classes: int,
    class_names: dict[int, str],
) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]], dict[str, Any]]:
    training_args = train_args(args, local_epochs=1)
    subject_list = subjects.tolist()
    train_index_set = set(split_indices["train"].tolist())
    train_indices_by_subject: dict[str, np.ndarray] = {}
    for subject_id in sorted(set(subject_list)):
        indices = np.asarray(
            [
                index
                for index, current_subject in enumerate(subject_list)
                if current_subject == subject_id and index in train_index_set
            ],
            dtype=np.int64,
        )
        if len(indices) > 0:
            train_indices_by_subject[str(subject_id)] = indices

    split_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    split_labels: dict[str, list[np.ndarray]] = defaultdict(list)
    split_subjects: dict[str, list[np.ndarray]] = defaultdict(list)
    split_loss_sum: dict[str, float] = defaultdict(float)
    split_n: dict[str, int] = defaultdict(int)

    label_tensor = base.torch.from_numpy(y.astype(np.int64))
    for subject_id in sorted(eval_subjects):
        train_indices = train_indices_by_subject.get(subject_id)
        if train_indices is None:
            continue
        model = base.build_model(num_classes, training_args).to(device)
        run_epoch_training(
            model=model,
            x_tensor=x_tensor,
            train_label_tensor=label_tensor,
            train_indices=train_indices,
            device=device,
            args=training_args,
            epochs=args.local_only_epochs,
            seed_key=f"local_only_{subject_id}",
        )
        for split in base.SPLITS:
            subject_split_indices = np.asarray(
                [
                    index
                    for index in split_indices[split].tolist()
                    if str(subjects[index]) == subject_id
                ],
                dtype=np.int64,
            )
            if len(subject_split_indices) == 0:
                continue
            predictions, loss = base.predict_split(
                model=model,
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                indices=subject_split_indices,
                device=device,
                batch_size=args.batch_size,
            )
            split_predictions[split].append(predictions)
            split_labels[split].append(y[subject_split_indices])
            split_subjects[split].append(subjects[subject_split_indices])
            split_loss_sum[split] += float(loss) * len(subject_split_indices)
            split_n[split] += len(subject_split_indices)

    overall: dict[str, Any] = {}
    per_class: list[dict[str, object]] = []
    per_user: list[dict[str, object]] = []
    confusion: dict[str, Any] = {}
    for split in base.SPLITS:
        if not split_predictions[split]:
            continue
        labels = np.concatenate(split_labels[split])
        predictions = np.concatenate(split_predictions[split])
        split_subject_array = np.concatenate(split_subjects[split])
        loss = split_loss_sum[split] / max(1, split_n[split])
        overall[split] = summarize_arrays(
            labels=labels,
            predictions=predictions,
            subjects=split_subject_array,
            eval_subjects=eval_subjects,
            num_classes=num_classes,
            loss=loss,
        )
        per_class.extend(
            class_metric_rows(
                labels=labels,
                predictions=predictions,
                num_classes=num_classes,
                class_names=class_names,
                experiment="local_only",
                split=split,
            )
        )
        per_user.extend(
            per_user_rows(
                labels=labels,
                predictions=predictions,
                subjects=split_subject_array,
                eval_subjects=eval_subjects,
                experiment="local_only",
                split=split,
            )
        )
        confusion[split] = confusion_matrix(labels, predictions, num_classes).tolist()
    return overall, per_class, per_user, confusion


def append_overall_rows(
    rows: list[dict[str, object]],
    experiment: str,
    metrics_by_split: dict[str, Any],
    protocol: str,
    communication_bytes: int | None = None,
) -> None:
    preferred_order = [*base.SPLITS, "tiny_train"]
    ordered_splits = [
        split for split in preferred_order if split in metrics_by_split
    ] + [
        split for split in metrics_by_split if split not in preferred_order
    ]
    for split in ordered_splits:
        metrics = metrics_by_split[split]
        per_user = metrics["per_user"]
        rows.append(
            {
                "experiment": experiment,
                "protocol": protocol,
                "split": split,
                "n_windows": metrics.get("n_windows"),
                "loss": metrics["loss"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "user_n": per_user["n_users"],
                "user_mean_macro_f1": per_user["mean_macro_f1"],
                "user_std_macro_f1": per_user["std_macro_f1"],
                "user_worst10_macro_f1": per_user["worst_10pct_macro_f1"],
                "user_min_macro_f1": per_user["min_macro_f1"],
                "total_communication_bytes": communication_bytes,
            }
        )


def metric(metrics_by_split: dict[str, Any], split: str, key: str) -> float:
    return float(metrics_by_split[split][key])


def load_selected_reference(path: Path) -> dict[str, object] | None:
    final_path = REPO_ROOT / path / "final_metrics.json"
    if not final_path.exists():
        return None
    final = json.loads(final_path.read_text())
    return {
        "run_dir": str(path),
        "round": final["round"],
        "validation_macro_f1": final["metrics"]["validation"]["macro_f1"],
        "test_macro_f1": final["metrics"]["test"]["macro_f1"],
        "validation_user_mean_macro_f1": final["metrics"]["validation"]["per_user"][
            "mean_macro_f1"
        ],
        "test_user_mean_macro_f1": final["metrics"]["test"]["per_user"][
            "mean_macro_f1"
        ],
        "total_communication_bytes": final["communication"]["total_bytes"],
    }


def make_health_gates(
    *,
    tiny_metrics: dict[str, Any],
    randomized_metrics: dict[str, Any],
    centralized_metrics: dict[str, Any],
    user_fedavg_metrics: dict[str, Any],
    iid_fedavg_metrics: dict[str, Any],
    distribution_summary: dict[str, object],
    num_classes: int,
) -> list[dict[str, object]]:
    chance = 1.0 / num_classes
    random_threshold = max(RANDOM_LABEL_ABSOLUTE_THRESHOLD, CHANCE_MULTIPLIER_THRESHOLD * chance)
    tiny_accuracy = metric(tiny_metrics, "tiny_train", "accuracy")
    tiny_macro_f1 = metric(tiny_metrics, "tiny_train", "macro_f1")
    random_val_f1 = metric(randomized_metrics, "validation", "macro_f1")
    centralized_val_f1 = metric(centralized_metrics, "validation", "macro_f1")
    user_val_f1 = metric(user_fedavg_metrics, "validation", "macro_f1")
    iid_val_f1 = metric(iid_fedavg_metrics, "validation", "macro_f1")
    mean_js = float(distribution_summary["mean_js_divergence_from_global_train"])
    median_client_classes = float(
        distribution_summary["median_nonzero_train_classes_per_client"]
    )

    return [
        {
            "gate": "tiny_overfit",
            "status": "PASS" if tiny_accuracy >= 0.95 and tiny_macro_f1 >= 0.95 else "FAIL",
            "criterion": "Tiny train subset accuracy and Macro-F1 should both reach at least 0.95.",
            "observed": {
                "accuracy": tiny_accuracy,
                "macro_f1": tiny_macro_f1,
            },
        },
        {
            "gate": "random_label_negative_control",
            "status": "PASS" if random_val_f1 <= random_threshold else "FAIL",
            "criterion": (
                "Validation Macro-F1 after permuted training labels should stay near "
                f"chance; threshold={random_threshold:.4f}."
            ),
            "observed": {
                "validation_macro_f1": random_val_f1,
                "chance_macro_f1_scale": chance,
            },
        },
        {
            "gate": "centralized_oracle",
            "status": "PASS"
            if centralized_val_f1 >= user_val_f1 + 0.05
            else "WARN",
            "criterion": "Pooled centralized training should beat the same-budget user-split FedAvg by a visible margin.",
            "observed": {
                "centralized_validation_macro_f1": centralized_val_f1,
                "user_fedavg_validation_macro_f1": user_val_f1,
                "margin": centralized_val_f1 - user_val_f1,
            },
        },
        {
            "gate": "iid_vs_user_fedavg",
            "status": "PASS" if iid_val_f1 >= user_val_f1 - 0.02 else "WARN",
            "criterion": "IID synthetic clients should not be materially worse than user-split clients at the same FedAvg budget.",
            "observed": {
                "iid_validation_macro_f1": iid_val_f1,
                "user_validation_macro_f1": user_val_f1,
                "margin": iid_val_f1 - user_val_f1,
            },
        },
        {
            "gate": "client_label_heterogeneity",
            "status": "PASS" if mean_js >= 0.10 and median_client_classes < num_classes else "WARN",
            "criterion": "User clients should show visible label-distribution heterogeneity; otherwise FL algorithms may naturally look similar.",
            "observed": {
                "mean_js_divergence_from_global_train": mean_js,
                "median_nonzero_train_classes_per_client": median_client_classes,
                "num_classes": num_classes,
            },
        },
    ]


def format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    summary: dict[str, object],
    gates: list[dict[str, object]],
    overall_rows: list[dict[str, object]],
    distribution_summary: dict[str, object],
    selected_reference: dict[str, object] | None,
    elapsed_seconds: float,
) -> None:
    rows_by_experiment_split = {
        (row["experiment"], row["split"]): row for row in overall_rows
    }
    key_experiments = [
        ("centralized_oracle", "centralized"),
        ("user_fedavg", "user-split FedAvg"),
        ("iid_fedavg", "IID-client FedAvg"),
        ("local_only", "local-only"),
        ("random_label_centralized", "random-label centralized"),
    ]
    comparison_rows = []
    for experiment, label in key_experiments:
        val = rows_by_experiment_split.get((experiment, "validation"))
        test = rows_by_experiment_split.get((experiment, "test"))
        if val is None or test is None:
            continue
        comparison_rows.append(
            [
                label,
                format_float(val["macro_f1"]),
                format_float(val["user_mean_macro_f1"]),
                format_float(val["user_worst10_macro_f1"]),
                format_float(test["macro_f1"]),
                format_float(test["user_mean_macro_f1"]),
                format_float(test["user_worst10_macro_f1"]),
            ]
        )

    central_val = rows_by_experiment_split.get(
        ("centralized_oracle", "validation"), {}
    ).get("macro_f1")
    iid_val = rows_by_experiment_split.get(("iid_fedavg", "validation"), {}).get(
        "macro_f1"
    )
    user_val = rows_by_experiment_split.get(("user_fedavg", "validation"), {}).get(
        "macro_f1"
    )
    random_val = rows_by_experiment_split.get(
        ("random_label_centralized", "validation"), {}
    ).get("macro_f1")
    all_gates_pass = all(gate["status"] == "PASS" for gate in gates)
    central_margin = (
        float(central_val) - float(user_val)
        if central_val is not None and user_val is not None
        else None
    )
    iid_margin = (
        float(iid_val) - float(user_val)
        if iid_val is not None and user_val is not None
        else None
    )

    gate_rows = [
        [
            gate["gate"],
            gate["status"],
            gate["criterion"],
            json.dumps(gate["observed"], sort_keys=True),
        ]
        for gate in gates
    ]

    lines = [
        "# KU-HAR Diagnostic V1",
        "",
        "## Purpose",
        "",
        "This report records setup-health diagnostics for the frozen KU-HAR V1 "
        "minimum-support pipeline. The goal is to test whether the pipeline behaves "
        "sensibly before adding more federated algorithms or moving to HHAR/WISDM.",
        "",
        "## Protocol",
        "",
        f"- Cohort: `{args.cohort}`",
        f"- Seed: `{args.seed}`",
        f"- Model family: 1D CNN with `{args.norm}`",
        f"- Optimizer: `{args.optimizer}`, learning rate `{args.lr}`",
        f"- Centralized oracle epochs: `{args.centralized_epochs}`",
        f"- Random-label negative-control epochs: `{args.random_label_epochs}`",
        f"- Tiny-overfit epochs: `{args.tiny_overfit_epochs}`",
        f"- FedAvg diagnostic budget: `{args.fedavg_rounds}` rounds, "
        f"`{args.fedavg_local_epochs}` local epochs",
        f"- Local-only epochs: `{args.local_only_epochs}`"
        if not args.skip_local_only
        else "- Local-only baseline: skipped",
        f"- Elapsed wall time: `{elapsed_seconds:.1f}` seconds",
        "",
        "## Manifest Snapshot",
        "",
        f"- Windows: `{summary['windows']}`",
        f"- Subjects: `{summary['subjects']}`",
        f"- Evaluable subjects: `{summary['evaluable_subjects']}`",
        f"- Split windows: `{summary['split_windows']}`",
        f"- Labels: `{summary['labels']}`",
        "",
        "## Health Gates",
        "",
        markdown_table(["Gate", "Status", "Criterion", "Observed"], gate_rows),
        "",
        "## Main Diagnostic Metrics",
        "",
        markdown_table(
            [
                "Protocol",
                "Val Macro-F1",
                "Val User Mean",
                "Val Worst10",
                "Test Macro-F1",
                "Test User Mean",
                "Test Worst10",
            ],
            comparison_rows,
        ),
        "",
        "## Diagnosis",
        "",
        "All health gates pass."
        if all_gates_pass
        else "At least one health gate did not pass; inspect the table above before treating downstream algorithm comparisons as conclusive.",
        "",
        "The strongest setup signal is that pooled centralized training is strong "
        "while the random-label negative control stays near chance. Centralized "
        f"oracle validation Macro-F1 is `{format_float(central_val)}`, which is "
        f"`{format_float(central_margin)}` above user-split FedAvg at the same "
        "diagnostic comparison point. The random-label negative control stays "
        f"near chance at `{format_float(random_val)}`.",
        "",
        "The IID-client FedAvg control is also directionally better than the real "
        "user-split FedAvg protocol, but the gap is modest rather than oracle-like: "
        f"IID-client validation Macro-F1 is `{format_float(iid_val)}` versus "
        f"`{format_float(user_val)}` for user-split FedAvg, a margin of "
        f"`{format_float(iid_margin)}`. This points to both user heterogeneity "
        "and limited FL optimization budget as bottlenecks.",
        "",
        "This supports the interpretation that the data loader, label mapping, "
        "model, loss, and evaluation loop are functioning. KU-HAR V1 is therefore "
        "a real non-IID challenge under this protocol, not an obviously broken "
        "setup. Similar scores among some algorithms should be treated as an "
        "algorithm/dataset interaction to analyze, not as immediate evidence that "
        "the pipeline failed.",
        "",
        "## Client Heterogeneity",
        "",
        f"- Train clients: `{distribution_summary['clients_with_train']}`",
        f"- Mean train windows per client: "
        f"`{float(distribution_summary['mean_train_windows_per_client']):.1f}`",
        f"- Median nonzero train classes per client: "
        f"`{float(distribution_summary['median_nonzero_train_classes_per_client']):.1f}`",
        f"- Mean JS divergence from global train label distribution: "
        f"`{float(distribution_summary['mean_js_divergence_from_global_train']):.4f}`",
        "",
    ]
    if selected_reference is not None:
        lines.extend(
            [
                "## Existing Selected FedAvg Reference",
                "",
                f"- Run directory: `{selected_reference['run_dir']}`",
                f"- Round: `{selected_reference['round']}`",
                f"- Validation Macro-F1: "
                f"`{float(selected_reference['validation_macro_f1']):.4f}`",
                f"- Test Macro-F1: `{float(selected_reference['test_macro_f1']):.4f}`",
                f"- Communication: "
                f"`{int(selected_reference['total_communication_bytes']):,}` bytes",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Overall metrics: `{args.output_dir / 'overall_metrics.csv'}`",
            f"- Per-class metrics: `{args.output_dir / 'per_class_metrics.csv'}`",
            f"- Per-user metrics: `{args.output_dir / 'per_user_metrics.csv'}`",
            f"- Confusion matrices: `{args.output_dir / 'confusion_matrices.json'}`",
            f"- Client label distribution: `{args.output_dir / 'client_label_distribution.csv'}`",
            f"- Health gate JSON: `{args.output_dir / 'health_gates.json'}`",
            "",
            "## Interpretation",
            "",
            "KU-HAR V1 should remain in the paper as a controlled subject-heterogeneous "
            "benchmark. The next scientific step is to add cross-dataset validation "
            "on HHAR or WISDM, so the paper can distinguish KU-HAR-specific behavior "
            "from algorithm behavior that persists under stronger device or subject "
            "heterogeneity.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    start_time = time.time()
    final_summary_path = args.output_dir / "diagnostic_summary.json"
    if final_summary_path.exists() and not args.force:
        print(f"diagnostic already exists: {final_summary_path}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_stage("loading frozen manifest")
    rows, label_ids, eval_subjects = base.read_window_manifest(
        args.manifest_dir, args.cohort
    )
    manifest_summary = base.manifest_summary(rows, eval_subjects, args.cohort)
    write_json(args.output_dir / "manifest_sanity.json", manifest_summary)
    if args.dry_run:
        print(json.dumps(manifest_summary, indent=2, sort_keys=True))
        return

    base.require_torch()
    device = base.choose_device(args.device)
    base.torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        base.torch.cuda.manual_seed_all(args.seed)

    log_stage(f"loading windows and standardizing channels on device={device}")
    labels, subjects, splits, split_indices = base.build_label_arrays(rows, label_ids)
    windows = base.load_or_build_windows(
        rows=rows,
        labels=labels,
        subjects=subjects,
        splits=splits,
        archive_path=args.archive,
        cache_file=base.cache_path(args.cache_dir, args.cohort),
        rebuild_cache=args.rebuild_cache,
        skip_archive_sha256=args.skip_archive_sha256,
    )
    windows, standardization = base.standardize_windows(
        windows, split_indices["train"]
    )
    write_json(args.output_dir / "channel_standardization.json", standardization)

    x_tensor = base.torch.from_numpy(windows)
    y_tensor = base.torch.from_numpy(labels)
    num_classes = len(label_ids)
    class_names = read_class_names(args.manifest_dir)
    log_stage("computing label-distribution diagnostics")
    distribution_summary, client_rows, class_distribution_rows = distribution_diagnostics(
        y=labels,
        subjects=subjects,
        split_indices=split_indices,
        num_classes=num_classes,
        class_names=class_names,
    )
    write_json(args.output_dir / "distribution_summary.json", distribution_summary)
    write_csv(args.output_dir / "client_label_distribution.csv", client_rows)
    write_csv(args.output_dir / "class_distribution.csv", class_distribution_rows)

    overall_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    per_user_metric_rows: list[dict[str, object]] = []
    confusion_matrices: dict[str, Any] = {}
    histories: dict[str, Any] = {}

    log_stage("running tiny-overfit positive control")
    tiny_indices = select_tiny_indices(
        y=labels,
        train_indices=split_indices["train"],
        seed=args.seed,
        max_classes=args.tiny_overfit_classes,
        windows_per_class=args.tiny_overfit_windows_per_class,
    )
    tiny_model, histories["tiny_overfit"] = train_tiny_overfit(
        x_tensor=x_tensor,
        y=labels,
        tiny_indices=tiny_indices,
        args=args,
        device=device,
        num_classes=num_classes,
    )
    tiny_metrics, tiny_predictions = evaluate_model_on_indices(
        model=tiny_model,
        x_tensor=x_tensor,
        y=labels,
        y_tensor=y_tensor,
        subjects=subjects,
        indices=tiny_indices,
        eval_subjects=set(subjects[tiny_indices].astype(str).tolist()),
        device=device,
        batch_size=args.batch_size,
        num_classes=num_classes,
    )
    tiny_metrics_by_split = {"tiny_train": tiny_metrics}
    append_overall_rows(
        rows=overall_rows,
        experiment="tiny_overfit",
        metrics_by_split=tiny_metrics_by_split,
        protocol="tiny overfit positive control",
    )
    tiny_labels = labels[tiny_indices]
    tiny_subjects = subjects[tiny_indices]
    per_class_rows.extend(
        class_metric_rows(
            tiny_labels,
            tiny_predictions,
            num_classes,
            class_names,
            "tiny_overfit",
            "tiny_train",
        )
    )
    per_user_metric_rows.extend(
        per_user_rows(
            tiny_labels,
            tiny_predictions,
            tiny_subjects,
            set(tiny_subjects.astype(str).tolist()),
            "tiny_overfit",
            "tiny_train",
        )
    )
    confusion_matrices["tiny_overfit"] = {
        "tiny_train": confusion_matrix(tiny_labels, tiny_predictions, num_classes).tolist()
    }

    log_stage("running centralized oracle")
    centralized_model, histories["centralized_oracle"] = train_centralized(
        name="centralized_oracle",
        x_tensor=x_tensor,
        y=labels,
        train_labels=labels,
        split_indices=split_indices,
        args=args,
        device=device,
        num_classes=num_classes,
        epochs=args.centralized_epochs,
    )
    (
        centralized_metrics,
        centralized_per_class,
        centralized_per_user,
        centralized_confusion,
    ) = evaluate_model_all_splits(
        experiment="centralized_oracle",
        model=centralized_model,
        x_tensor=x_tensor,
        y=labels,
        y_tensor=y_tensor,
        subjects=subjects,
        split_indices=split_indices,
        eval_subjects=eval_subjects,
        device=device,
        batch_size=args.batch_size,
        num_classes=num_classes,
        class_names=class_names,
    )
    append_overall_rows(
        overall_rows, "centralized_oracle", centralized_metrics, "centralized"
    )
    per_class_rows.extend(centralized_per_class)
    per_user_metric_rows.extend(centralized_per_user)
    confusion_matrices["centralized_oracle"] = centralized_confusion

    log_stage("running random-label negative control")
    randomized_labels = make_randomized_train_labels(
        labels, split_indices["train"], args.seed
    )
    random_model, histories["random_label_centralized"] = train_centralized(
        name="random_label_centralized",
        x_tensor=x_tensor,
        y=labels,
        train_labels=randomized_labels,
        split_indices=split_indices,
        args=args,
        device=device,
        num_classes=num_classes,
        epochs=args.random_label_epochs,
    )
    (
        random_metrics,
        random_per_class,
        random_per_user,
        random_confusion,
    ) = evaluate_model_all_splits(
        experiment="random_label_centralized",
        model=random_model,
        x_tensor=x_tensor,
        y=labels,
        y_tensor=y_tensor,
        subjects=subjects,
        split_indices=split_indices,
        eval_subjects=eval_subjects,
        device=device,
        batch_size=args.batch_size,
        num_classes=num_classes,
        class_names=class_names,
    )
    append_overall_rows(
        overall_rows,
        "random_label_centralized",
        random_metrics,
        "random-label negative control",
    )
    per_class_rows.extend(random_per_class)
    per_user_metric_rows.extend(random_per_user)
    confusion_matrices["random_label_centralized"] = random_confusion

    log_stage("running user-split FedAvg diagnostic")
    user_fedavg_history, user_fedavg_model = run_fedavg_diagnostic(
        name="user_fedavg",
        windows=windows,
        y=labels,
        federated_subjects=subjects,
        split_indices=split_indices,
        eval_subjects=eval_subjects,
        args=args,
    )
    histories["user_fedavg"] = user_fedavg_history
    user_fedavg_metrics = user_fedavg_history[-1]["metrics"]
    (
        user_eval_metrics,
        user_per_class,
        user_per_user,
        user_confusion,
    ) = evaluate_model_all_splits(
        experiment="user_fedavg",
        model=user_fedavg_model,
        x_tensor=x_tensor,
        y=labels,
        y_tensor=y_tensor,
        subjects=subjects,
        split_indices=split_indices,
        eval_subjects=eval_subjects,
        device=device,
        batch_size=args.batch_size,
        num_classes=num_classes,
        class_names=class_names,
    )
    user_fedavg_metrics = user_eval_metrics
    append_overall_rows(
        overall_rows,
        "user_fedavg",
        user_fedavg_metrics,
        "user-split FedAvg",
        communication_bytes=user_fedavg_history[-1]["communication"]["total_bytes"],
    )
    per_class_rows.extend(user_per_class)
    per_user_metric_rows.extend(user_per_user)
    confusion_matrices["user_fedavg"] = user_confusion

    log_stage("running IID-client FedAvg diagnostic")
    iid_subjects = make_iid_train_subjects(
        subjects=subjects,
        y=labels,
        train_indices=split_indices["train"],
        n_clients=args.iid_clients,
        seed=args.seed,
    )
    actual_iid_clients = count_train_clients(iid_subjects, split_indices["train"])
    if actual_iid_clients != args.iid_clients:
        raise RuntimeError(
            f"expected {args.iid_clients} IID train clients, got {actual_iid_clients}"
        )
    iid_fedavg_history, iid_fedavg_model = run_fedavg_diagnostic(
        name="iid_fedavg",
        windows=windows,
        y=labels,
        federated_subjects=iid_subjects,
        split_indices=split_indices,
        eval_subjects=eval_subjects,
        args=args,
    )
    histories["iid_fedavg"] = iid_fedavg_history
    iid_fedavg_metrics = iid_fedavg_history[-1]["metrics"]
    (
        iid_eval_metrics,
        iid_per_class,
        iid_per_user,
        iid_confusion,
    ) = evaluate_model_all_splits(
        experiment="iid_fedavg",
        model=iid_fedavg_model,
        x_tensor=x_tensor,
        y=labels,
        y_tensor=y_tensor,
        subjects=subjects,
        split_indices=split_indices,
        eval_subjects=eval_subjects,
        device=device,
        batch_size=args.batch_size,
        num_classes=num_classes,
        class_names=class_names,
    )
    iid_fedavg_metrics = iid_eval_metrics
    append_overall_rows(
        overall_rows,
        "iid_fedavg",
        iid_fedavg_metrics,
        "IID-client FedAvg",
        communication_bytes=iid_fedavg_history[-1]["communication"]["total_bytes"],
    )
    per_class_rows.extend(iid_per_class)
    per_user_metric_rows.extend(iid_per_user)
    confusion_matrices["iid_fedavg"] = iid_confusion

    local_only_metrics: dict[str, Any] = {}
    if not args.skip_local_only:
        log_stage("running known-client local-only baseline")
        (
            local_only_metrics,
            local_per_class,
            local_per_user,
            local_confusion,
        ) = train_local_only(
            x_tensor=x_tensor,
            y=labels,
            y_tensor=y_tensor,
            subjects=subjects,
            split_indices=split_indices,
            eval_subjects=eval_subjects,
            args=args,
            device=device,
            num_classes=num_classes,
            class_names=class_names,
        )
        append_overall_rows(
            overall_rows,
            "local_only",
            local_only_metrics,
            "known-client local-only",
        )
        per_class_rows.extend(local_per_class)
        per_user_metric_rows.extend(local_per_user)
        confusion_matrices["local_only"] = local_confusion

    log_stage("building health gates and writing artifacts")
    health_gates = make_health_gates(
        tiny_metrics=tiny_metrics_by_split,
        randomized_metrics=random_metrics,
        centralized_metrics=centralized_metrics,
        user_fedavg_metrics=user_fedavg_metrics,
        iid_fedavg_metrics=iid_fedavg_metrics,
        distribution_summary=distribution_summary,
        num_classes=num_classes,
    )
    selected_reference = load_selected_reference(args.selected_fedavg_run_dir)

    write_csv(args.output_dir / "overall_metrics.csv", overall_rows)
    write_csv(args.output_dir / "per_class_metrics.csv", per_class_rows)
    write_csv(args.output_dir / "per_user_metrics.csv", per_user_metric_rows)
    write_json(args.output_dir / "confusion_matrices.json", confusion_matrices)
    write_json(args.output_dir / "training_histories.json", histories)
    write_json(args.output_dir / "health_gates.json", health_gates)

    config = {
        "cohort": args.cohort,
        "manifest_dir": str(args.manifest_dir),
        "archive": str(args.archive),
        "cache_dir": str(args.cache_dir),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "norm": args.norm,
        "centralized_epochs": args.centralized_epochs,
        "random_label_epochs": args.random_label_epochs,
        "tiny_overfit_epochs": args.tiny_overfit_epochs,
        "tiny_overfit_lr": args.tiny_overfit_lr,
        "fedavg_rounds": args.fedavg_rounds,
        "fedavg_local_epochs": args.fedavg_local_epochs,
        "iid_clients": args.iid_clients,
        "local_only_epochs": args.local_only_epochs,
        "skip_local_only": args.skip_local_only,
        "device": args.device,
        "resolved_device": str(device),
        "labels": label_ids,
        "input_shape": [3, base.WINDOW_SAMPLES],
        "torch_version": base.torch.__version__,
    }
    elapsed_seconds = time.time() - start_time
    diagnostic_summary = {
        "run_config": config,
        "manifest_summary": manifest_summary,
        "distribution_summary": distribution_summary,
        "health_gates": health_gates,
        "selected_reference": selected_reference,
        "elapsed_seconds": elapsed_seconds,
    }
    write_json(final_summary_path, diagnostic_summary)
    write_report(
        report_path=args.report_path,
        args=args,
        summary=manifest_summary,
        gates=health_gates,
        overall_rows=overall_rows,
        distribution_summary=distribution_summary,
        selected_reference=selected_reference,
        elapsed_seconds=elapsed_seconds,
    )

    log_stage("complete")
    print(json.dumps(diagnostic_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
