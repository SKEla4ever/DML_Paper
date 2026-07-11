#!/usr/bin/env python3
"""1D CNN FedAvg baseline for the frozen KU-HAR V1 manifest."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by environment
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


EXPECTED_ARCHIVE_SHA256 = (
    "9fe5d0052f2f1d6711afac42ee4badd968116afa8ba4b8ba591f4fdd771c2ec2"
)
DEFAULT_MANIFEST_DIR = Path("kuhar_delivery/data/processed/kuhar")
DEFAULT_ARCHIVE = Path("kuhar_delivery/data/raw/kuhar/2.Trimmed_interpolated_data.zip")
DEFAULT_OUTPUT_DIR = Path("outputs/kuhar_1d_cnn_fedavg_v1")
DEFAULT_CACHE_DIR = Path("outputs/cache")
WINDOW_SAMPLES = 300
SPLITS = ("train", "validation", "test")
SUPPORTED_TEST_CLASS_WINDOWS = 3


@dataclass(frozen=True)
class WindowRow:
    recording_id: str
    subject_id: str
    label_id: int
    split: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a neural 1D CNN FedAvg baseline on KU-HAR frozen V1."
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--cohort",
        choices=("minimum_support", "full_sparse"),
        default="minimum_support",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument(
        "--norm",
        choices=("batchnorm", "groupnorm", "none"),
        default="batchnorm",
        help="Normalization layer used after each Conv1d block.",
    )
    parser.add_argument("--groupnorm-groups", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-archive-sha256", action="store_true")
    return parser.parse_args()


def require_torch() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise SystemExit(
            "PyTorch is required for algorithms/1d_cnn_fedavg. "
            "Install dependencies from algorithms/1d_cnn_fedavg/requirements.txt."
        ) from TORCH_IMPORT_ERROR


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:16], 16)


def read_subject_summary(manifest_dir: Path) -> dict[str, dict[str, str]]:
    path = manifest_dir / "kuhar_subject_summary.csv"
    with path.open(newline="") as handle:
        return {row["subject_id"]: row for row in csv.DictReader(handle)}


def read_window_manifest(
    manifest_dir: Path, cohort: str
) -> tuple[list[WindowRow], list[int], set[str]]:
    subject_summary = read_subject_summary(manifest_dir)
    if cohort == "minimum_support":
        label_ids = list(range(17))
        eval_subjects = {
            subject_id
            for subject_id, row in subject_summary.items()
            if row["minimum_support_client"] == "True"
        }
    else:
        label_ids = list(range(18))
        eval_subjects = {
            subject_id
            for subject_id, row in subject_summary.items()
            if row["realized_evaluable_client"] == "True"
        }

    rows: list[WindowRow] = []
    manifest_path = manifest_dir / "kuhar_window_split_manifest.csv.gz"
    with gzip.open(manifest_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            subject_id = row["subject_id"]
            label_id = int(row["activity_id"])
            if cohort == "minimum_support":
                subject = subject_summary.get(subject_id)
                if not subject or subject["minimum_support_client"] != "True":
                    continue
                if label_id == 17:
                    continue
            rows.append(
                WindowRow(
                    recording_id=row["recording_id"],
                    subject_id=subject_id,
                    label_id=label_id,
                    split=row["split"],
                    start=int(row["start_sample_offset"]),
                    end=int(row["end_sample_offset_exclusive"]),
                )
            )

    observed_labels = sorted({row.label_id for row in rows})
    missing = sorted(set(label_ids) - set(observed_labels))
    if missing:
        raise RuntimeError(f"cohort {cohort} is missing label IDs {missing}")
    return rows, label_ids, eval_subjects


def manifest_summary(
    rows: list[WindowRow], eval_subjects: set[str], cohort: str
) -> dict[str, object]:
    by_split = Counter(row.split for row in rows)
    by_subject_split: dict[str, Counter] = defaultdict(Counter)
    by_label_split: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        by_subject_split[row.subject_id][row.split] += 1
        by_label_split[row.label_id][row.split] += 1
    return {
        "cohort": cohort,
        "windows": len(rows),
        "subjects": len(by_subject_split),
        "evaluable_subjects": len(eval_subjects),
        "split_windows": {split: by_split[split] for split in SPLITS},
        "subjects_with_train": sum(
            1 for counts in by_subject_split.values() if counts["train"] > 0
        ),
        "labels": sorted(by_label_split),
        "label_split_windows": {
            str(label): {split: counts[split] for split in SPLITS}
            for label, counts in sorted(by_label_split.items())
        },
    }


def build_label_arrays(
    rows: list[WindowRow], label_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    label_to_position = {label_id: index for index, label_id in enumerate(label_ids)}
    labels = np.asarray([label_to_position[row.label_id] for row in rows], dtype=np.int64)
    subjects = np.asarray([row.subject_id for row in rows])
    splits = np.asarray([row.split for row in rows])
    split_indices = {
        split: np.where(splits == split)[0].astype(np.int64) for split in SPLITS
    }
    return labels, subjects, splits, split_indices


def cache_path(cache_dir: Path, cohort: str) -> Path:
    return cache_dir / f"kuhar_v1_{cohort}_raw_accel_windows.npz"


def load_or_build_windows(
    rows: list[WindowRow],
    labels: np.ndarray,
    subjects: np.ndarray,
    splits: np.ndarray,
    archive_path: Path,
    cache_file: Path,
    rebuild_cache: bool,
    skip_archive_sha256: bool,
) -> np.ndarray:
    if cache_file.exists() and not rebuild_cache:
        with np.load(cache_file, allow_pickle=False) as cached:
            return cached["windows"].astype(np.float32)

    if not archive_path.exists():
        raise FileNotFoundError(
            "KU-HAR raw archive not found. Expected "
            f"{archive_path}. Place 2.Trimmed_interpolated_data.zip there first."
        )
    if not skip_archive_sha256:
        actual_sha256 = file_sha256(archive_path)
        if actual_sha256 != EXPECTED_ARCHIVE_SHA256:
            raise RuntimeError(
                "archive SHA-256 does not match frozen KU-HAR V1 source: "
                f"{actual_sha256}"
            )

    rows_by_recording: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        rows_by_recording[row.recording_id].append(index)

    windows = np.empty((len(rows), 3, WINDOW_SAMPLES), dtype=np.float32)
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC validation failed at {bad_member}")
        for recording_id in sorted(rows_by_recording):
            with archive.open(recording_id) as handle:
                accel = np.loadtxt(
                    handle,
                    delimiter=",",
                    dtype=np.float32,
                    usecols=(1, 2, 3),
                    ndmin=2,
                )
            for index in rows_by_recording[recording_id]:
                row = rows[index]
                window = accel[row.start : row.end]
                if window.shape != (WINDOW_SAMPLES, 3):
                    raise ValueError(
                        f"{row.recording_id} window {row.start}:{row.end} "
                        f"has shape {window.shape}"
                    )
                windows[index] = window.T

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        windows=windows,
        labels=labels,
        subjects=subjects,
        splits=splits,
    )
    return windows


def standardize_windows(
    windows: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, list[float]]]:
    mean = windows[train_indices].mean(axis=(0, 2), keepdims=True)
    std = windows[train_indices].std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    standardized = (windows - mean) / std
    return standardized.astype(np.float32), {
        "channel_mean": mean.reshape(-1).astype(float).tolist(),
        "channel_std": std.reshape(-1).astype(float).tolist(),
    }


def choose_device(requested: str):
    require_torch()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalization_layer(channels: int, args: argparse.Namespace):
    if args.norm == "batchnorm":
        return nn.BatchNorm1d(channels)
    if args.norm == "groupnorm":
        groups = min(args.groupnorm_groups, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    return nn.Identity()


def build_model(num_classes: int, args: argparse.Namespace):
    require_torch()
    return nn.Sequential(
        nn.Conv1d(3, 32, kernel_size=7, padding=3, bias=False),
        normalization_layer(32, args),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
        normalization_layer(64, args),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
        normalization_layer(64, args),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(p=0.1),
        nn.Linear(64, num_classes),
    )


def state_payload_bytes(model) -> int:
    return int(
        sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())
    )


def learning_rate_for_round(args: argparse.Namespace, round_index: int) -> float:
    schedule = getattr(args, "lr_schedule", "constant")
    if schedule == "constant" or round_index <= 0:
        return float(args.lr)
    if schedule == "step":
        step_rounds = tuple(getattr(args, "lr_step_rounds", ()))
        if not step_rounds:
            raise ValueError("step LR schedule requires at least one step round")
        gamma = float(getattr(args, "lr_step_gamma", 0.1))
        decays = sum(round_index > step_round for step_round in step_rounds)
        return float(args.lr) * gamma**decays
    if schedule == "cosine":
        minimum = float(getattr(args, "lr_min", 0.0))
        denominator = max(int(args.rounds) - 1, 1)
        progress = min(max((round_index - 1) / denominator, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum + (float(args.lr) - minimum) * cosine
    raise ValueError(f"unknown LR schedule: {schedule}")


def make_optimizer(
    model, args: argparse.Namespace, learning_rate: float | None = None
):
    lr = float(args.lr) if learning_rate is None else learning_rate
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=args.weight_decay
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def train_local_model(
    global_state: dict[str, object],
    x_tensor,
    y_tensor,
    client_indices: np.ndarray,
    num_classes: int,
    device,
    args: argparse.Namespace,
    round_index: int,
    client_id: str,
) -> dict[str, object]:
    model = build_model(num_classes, args).to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = make_optimizer(
        model, args, learning_rate_for_round(args, round_index)
    )
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(stable_int(args.seed, round_index, client_id))
    for _epoch in range(args.local_epochs):
        shuffled = client_indices.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            batch_x = x_tensor[batch_indices].to(device)
            batch_y = y_tensor[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def average_state_dicts(
    state_dicts: list[dict[str, object]], example_counts: list[int]
) -> dict[str, object]:
    total_examples = float(sum(example_counts))
    averaged: dict[str, object] = {}
    for key in state_dicts[0]:
        first = state_dicts[0][key]
        if torch.is_floating_point(first):
            accumulator = torch.zeros_like(first, dtype=torch.float32)
            for state, count in zip(state_dicts, example_counts):
                accumulator += state[key].to(dtype=torch.float32) * (count / total_examples)
            averaged[key] = accumulator.to(dtype=first.dtype)
        else:
            averaged[key] = first.clone()
    return averaged


def macro_f1(labels: np.ndarray, predictions: np.ndarray, class_indices: list[int]) -> float:
    f1_values = []
    for class_index in class_indices:
        true_positive = int(
            np.sum((labels == class_index) & (predictions == class_index))
        )
        false_positive = int(
            np.sum((labels != class_index) & (predictions == class_index))
        )
        false_negative = int(
            np.sum((labels == class_index) & (predictions != class_index))
        )
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator == 0:
            continue
        f1_values.append(2 * true_positive / denominator)
    return float(np.mean(f1_values)) if f1_values else float("nan")


def per_user_macro_f1_summary(
    labels: np.ndarray,
    predictions: np.ndarray,
    subjects: np.ndarray,
    eval_subjects: set[str],
    supported_threshold: int,
) -> dict[str, float | int | None]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, subject_id in enumerate(subjects.tolist()):
        grouped[subject_id].append(index)

    values: list[float] = []
    for subject_id in sorted(eval_subjects):
        indices = grouped.get(subject_id, [])
        if not indices:
            continue
        subject_labels = labels[indices]
        subject_predictions = predictions[indices]
        supported_classes = [
            int(class_index)
            for class_index, count in Counter(subject_labels.tolist()).items()
            if count >= supported_threshold
        ]
        if len(supported_classes) < 3:
            continue
        values.append(
            macro_f1(subject_labels, subject_predictions, sorted(supported_classes))
        )

    if not values:
        return {
            "n_users": 0,
            "mean_macro_f1": None,
            "std_macro_f1": None,
            "worst_10pct_macro_f1": None,
            "min_macro_f1": None,
        }
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    worst_count = max(1, int(math.ceil(0.10 * ordered.shape[0])))
    return {
        "n_users": int(ordered.shape[0]),
        "mean_macro_f1": float(np.mean(ordered)),
        "std_macro_f1": float(np.std(ordered)),
        "worst_10pct_macro_f1": float(np.mean(ordered[:worst_count])),
        "min_macro_f1": float(ordered[0]),
    }


def predict_split(model, x_tensor, y_tensor, indices: np.ndarray, device, batch_size: int):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    predictions: list[np.ndarray] = []
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_x = x_tensor[batch_indices].to(device)
            batch_y = y_tensor[batch_indices].to(device)
            logits = model(batch_x)
            total_loss += float(criterion(logits, batch_y).detach().cpu())
            predictions.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(predictions), total_loss / len(indices)


def evaluate_split(
    model,
    x_tensor,
    y: np.ndarray,
    y_tensor,
    subjects: np.ndarray,
    indices: np.ndarray,
    eval_subjects: set[str],
    device,
    batch_size: int,
    num_classes: int,
) -> dict[str, object]:
    predictions, loss = predict_split(model, x_tensor, y_tensor, indices, device, batch_size)
    labels = y[indices]
    split_subjects = subjects[indices]
    return {
        "loss": float(loss),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": macro_f1(labels, predictions, list(range(num_classes))),
        "per_user": per_user_macro_f1_summary(
            labels=labels,
            predictions=predictions,
            subjects=split_subjects,
            eval_subjects=eval_subjects,
            supported_threshold=SUPPORTED_TEST_CLASS_WINDOWS,
        ),
    }


def evaluate_all_splits(
    model,
    x_tensor,
    y: np.ndarray,
    y_tensor,
    subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
    eval_subjects: set[str],
    device,
    batch_size: int,
    num_classes: int,
    evaluation_splits: tuple[str, ...] = SPLITS,
) -> dict[str, object]:
    return {
        split: evaluate_split(
            model=model,
            x_tensor=x_tensor,
            y=y,
            y_tensor=y_tensor,
            subjects=subjects,
            indices=split_indices[split],
            eval_subjects=eval_subjects,
            device=device,
            batch_size=batch_size,
            num_classes=num_classes,
        )
        for split in evaluation_splits
    }


def flatten_round_metrics(record: dict[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {
        "round": record["round"],
        "selected_clients": record["selected_clients"],
        "total_communication_bytes": record["communication"]["total_bytes"],
        "uplink_bytes": record["communication"]["uplink_bytes"],
        "downlink_bytes": record["communication"]["downlink_bytes"],
    }
    if "learning_rate" in record:
        flat["learning_rate"] = record["learning_rate"]
    metrics = record["metrics"]
    for split in metrics:
        split_metrics = metrics[split]
        flat[f"{split}_loss"] = split_metrics["loss"]
        flat[f"{split}_accuracy"] = split_metrics["accuracy"]
        flat[f"{split}_macro_f1"] = split_metrics["macro_f1"]
        per_user = split_metrics["per_user"]
        flat[f"{split}_user_mean_macro_f1"] = per_user["mean_macro_f1"]
        flat[f"{split}_user_worst10_macro_f1"] = per_user["worst_10pct_macro_f1"]
        flat[f"{split}_user_n"] = per_user["n_users"]
    return flat


def write_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_metrics_csv(path: Path, history: list[dict[str, object]]) -> None:
    rows = [flatten_round_metrics(record) for record in history]
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_fedavg(
    windows: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
    eval_subjects: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], object]:
    require_torch()
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    x_tensor = torch.from_numpy(windows)
    y_tensor = torch.from_numpy(y)
    train_labels = y[split_indices["train"]]
    num_classes = int(train_labels.max()) + 1
    global_model = build_model(num_classes, args).to(device)
    parameter_bytes = state_payload_bytes(global_model)
    parameter_count = int(sum(p.numel() for p in global_model.parameters()))

    train_indices_set = set(split_indices["train"].tolist())
    train_indices_by_client: dict[str, np.ndarray] = {}
    for subject_id in sorted(set(subjects.tolist())):
        indices = np.asarray(
            [
                index
                for index, current_subject in enumerate(subjects.tolist())
                if current_subject == subject_id and index in train_indices_set
            ],
            dtype=np.int64,
        )
        if len(indices) > 0:
            train_indices_by_client[subject_id] = indices

    clients = sorted(train_indices_by_client)
    selected_count = max(1, int(math.ceil(args.client_fraction * len(clients))))
    selected_count = min(selected_count, len(clients))
    rng = np.random.default_rng(args.seed)
    communication = {
        "parameter_count": parameter_count,
        "bytes_per_model_state": parameter_bytes,
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "total_bytes": 0,
    }
    history: list[dict[str, object]] = []
    evaluation_splits = tuple(getattr(args, "evaluation_splits", SPLITS))
    if not evaluation_splits or not set(evaluation_splits).issubset(SPLITS):
        raise ValueError(
            f"evaluation_splits must be a non-empty subset of {SPLITS}"
        )

    def record_metrics(round_index: int, selected_clients: int) -> None:
        metrics = evaluate_all_splits(
            model=global_model,
            x_tensor=x_tensor,
            y=y,
            y_tensor=y_tensor,
            subjects=subjects,
            split_indices=split_indices,
            eval_subjects=eval_subjects,
            device=device,
            batch_size=args.batch_size,
            num_classes=num_classes,
            evaluation_splits=evaluation_splits,
        )
        history.append(
            {
                "round": round_index,
                "learning_rate": learning_rate_for_round(args, round_index),
                "selected_clients": selected_clients,
                "communication": dict(communication),
                "metrics": metrics,
            }
        )

    record_metrics(0, 0)
    for round_index in range(1, args.rounds + 1):
        selected_clients = rng.choice(clients, size=selected_count, replace=False)
        global_state = copy.deepcopy(global_model.state_dict())
        local_states = []
        example_counts = []
        for client_id in selected_clients:
            client_id = str(client_id)
            indices = train_indices_by_client[client_id]
            local_states.append(
                train_local_model(
                    global_state=global_state,
                    x_tensor=x_tensor,
                    y_tensor=y_tensor,
                    client_indices=indices,
                    num_classes=num_classes,
                    device=device,
                    args=args,
                    round_index=round_index,
                    client_id=client_id,
                )
            )
            example_counts.append(int(len(indices)))

        averaged_state = average_state_dicts(local_states, example_counts)
        global_model.load_state_dict(averaged_state)
        communication["uplink_bytes"] += selected_count * parameter_bytes
        communication["downlink_bytes"] += selected_count * parameter_bytes
        communication["total_bytes"] = (
            communication["uplink_bytes"] + communication["downlink_bytes"]
        )
        if round_index % args.eval_every == 0 or round_index == args.rounds:
            record_metrics(round_index, selected_count)

    return history, global_model


def main() -> None:
    args = parse_args()
    start_time = time.time()
    if not (0 < args.client_fraction <= 1):
        raise ValueError("--client-fraction must be in (0, 1]")
    if args.rounds < 0:
        raise ValueError("--rounds must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, label_ids, eval_subjects = read_window_manifest(args.manifest_dir, args.cohort)
    summary = manifest_summary(rows, eval_subjects, args.cohort)
    write_json(args.output_dir / "manifest_sanity.json", summary)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    require_torch()
    labels, subjects, splits, split_indices = build_label_arrays(rows, label_ids)
    windows = load_or_build_windows(
        rows=rows,
        labels=labels,
        subjects=subjects,
        splits=splits,
        archive_path=args.archive,
        cache_file=cache_path(args.cache_dir, args.cohort),
        rebuild_cache=args.rebuild_cache,
        skip_archive_sha256=args.skip_archive_sha256,
    )
    windows, standardization = standardize_windows(windows, split_indices["train"])

    config = {
        "cohort": args.cohort,
        "manifest_dir": str(args.manifest_dir),
        "archive": str(args.archive),
        "cache_dir": str(args.cache_dir),
        "rounds": args.rounds,
        "client_fraction": args.client_fraction,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "norm": args.norm,
        "groupnorm_groups": args.groupnorm_groups,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "device": args.device,
        "labels": label_ids,
        "input_shape": [3, WINDOW_SAMPLES],
        "torch_version": torch.__version__,
    }
    write_json(args.output_dir / "run_config.json", config)
    write_json(args.output_dir / "channel_standardization.json", standardization)

    history, model = train_fedavg(
        windows=windows,
        y=labels,
        subjects=subjects,
        split_indices=split_indices,
        eval_subjects=eval_subjects,
        args=args,
    )
    write_json(args.output_dir / "metrics_history.json", history)
    write_json(args.output_dir / "final_metrics.json", history[-1])
    write_metrics_csv(args.output_dir / "round_metrics.csv", history)
    torch.save(model.state_dict(), args.output_dir / "final_model.pt")

    final = flatten_round_metrics(history[-1])
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "elapsed_seconds": time.time() - start_time,
                "final_round": final,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
