#!/usr/bin/env python3
"""FedAvg softmax sanity baseline for the frozen KU-HAR V1 manifest.

This script is intentionally lightweight: it depends only on NumPy plus the
Python standard library. It is a first training-pipeline checkpoint, not the
final neural HAR model.
"""

from __future__ import annotations

import argparse
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


EXPECTED_ARCHIVE_SHA256 = (
    "9fe5d0052f2f1d6711afac42ee4badd968116afa8ba4b8ba591f4fdd771c2ec2"
)
DEFAULT_MANIFEST_DIR = Path("kuhar_delivery/data/processed/kuhar")
DEFAULT_ARCHIVE = Path("kuhar_delivery/data/raw/kuhar/2.Trimmed_interpolated_data.zip")
DEFAULT_OUTPUT_DIR = Path("outputs/kuhar_fedavg_v1")
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
        description=(
            "Train a NumPy FedAvg softmax sanity baseline from the frozen "
            "KU-HAR V1 split manifest."
        )
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cohort",
        choices=("minimum_support", "full_sparse"),
        default="minimum_support",
        help="Use KU-HAR minimum-support main cohort or full sparse cohort.",
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests and write a manifest sanity summary only.",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help=(
            "Use deterministic synthetic features derived from labels and "
            "subjects. This verifies the training loop only; it is not an "
            "experiment result."
        ),
    )
    parser.add_argument(
        "--skip-archive-sha256",
        action="store_true",
        help="Skip the frozen source archive SHA-256 check.",
    )
    return parser.parse_args()


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
) -> tuple[list[WindowRow], list[int], set[str], dict[str, dict[str, str]]]:
    subject_summary = read_subject_summary(manifest_dir)
    eval_subjects: set[str] = set()
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
    return rows, label_ids, eval_subjects, subject_summary


def manifest_summary(
    rows: list[WindowRow], eval_subjects: set[str], cohort: str
) -> dict[str, object]:
    by_split = Counter(row.split for row in rows)
    by_label_split: dict[int, Counter] = defaultdict(Counter)
    by_subject_split: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_label_split[row.label_id][row.split] += 1
        by_subject_split[row.subject_id][row.split] += 1
    return {
        "cohort": cohort,
        "windows": len(rows),
        "split_windows": {split: by_split[split] for split in SPLITS},
        "subjects": len(by_subject_split),
        "evaluable_subjects": len(eval_subjects),
        "labels": sorted(by_label_split),
        "label_split_windows": {
            str(label_id): {split: counts[split] for split in SPLITS}
            for label_id, counts in sorted(by_label_split.items())
        },
        "subjects_with_train": sum(
            1 for counts in by_subject_split.values() if counts["train"] > 0
        ),
    }


def extract_feature_vector(window: np.ndarray) -> np.ndarray:
    if window.shape != (WINDOW_SAMPLES, 3):
        raise ValueError(f"expected {(WINDOW_SAMPLES, 3)}, got {window.shape}")
    features: list[float] = []
    for axis in range(3):
        values = window[:, axis]
        features.extend(
            [
                float(np.mean(values)),
                float(np.std(values)),
                float(np.min(values)),
                float(np.max(values)),
                float(np.percentile(values, 25)),
                float(np.percentile(values, 75)),
                float(np.sqrt(np.mean(values * values))),
                float(np.mean(np.abs(np.diff(values)))),
            ]
        )

    magnitude = np.sqrt(np.sum(window * window, axis=1))
    features.extend(
        [
            float(np.mean(magnitude)),
            float(np.std(magnitude)),
            float(np.min(magnitude)),
            float(np.max(magnitude)),
            float(np.sqrt(np.mean(magnitude * magnitude))),
        ]
    )

    for left, right in ((0, 1), (0, 2), (1, 2)):
        left_values = window[:, left]
        right_values = window[:, right]
        if np.std(left_values) < 1e-8 or np.std(right_values) < 1e-8:
            features.append(0.0)
        else:
            features.append(float(np.corrcoef(left_values, right_values)[0, 1]))
    return np.asarray(features, dtype=np.float32)


def load_archive_features(
    rows: list[WindowRow], archive_path: Path, skip_archive_sha256: bool
) -> np.ndarray:
    if not archive_path.exists():
        raise FileNotFoundError(
            "KU-HAR raw archive not found. Expected "
            f"{archive_path}. Place 2.Trimmed_interpolated_data.zip there, "
            "or run with --synthetic-smoke only for pipeline smoke testing."
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

    features_by_index: list[np.ndarray | None] = [None] * len(rows)
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
                features_by_index[index] = extract_feature_vector(window)

    if any(feature is None for feature in features_by_index):
        raise RuntimeError("internal feature extraction error")
    return np.vstack([feature for feature in features_by_index if feature is not None])


def make_synthetic_features(
    rows: list[WindowRow], label_ids: list[int], seed: int, feature_dim: int = 32
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    label_to_position = {label_id: index for index, label_id in enumerate(label_ids)}
    centers = rng.normal(0.0, 1.0, size=(len(label_ids), feature_dim)).astype(np.float32)
    subject_offsets: dict[str, np.ndarray] = {}
    features = np.empty((len(rows), feature_dim), dtype=np.float32)
    for index, row in enumerate(rows):
        if row.subject_id not in subject_offsets:
            subject_rng = np.random.default_rng(stable_int(seed, row.subject_id))
            subject_offsets[row.subject_id] = subject_rng.normal(
                0.0, 0.35, size=feature_dim
            ).astype(np.float32)
        noise_rng = np.random.default_rng(
            stable_int(seed, row.recording_id, row.start, row.split)
        )
        noise = noise_rng.normal(0.0, 0.8, size=feature_dim).astype(np.float32)
        features[index] = (
            centers[label_to_position[row.label_id]]
            + subject_offsets[row.subject_id]
            + noise
        )
    return features


def build_arrays(
    rows: list[WindowRow], label_ids: list[int]
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray], dict[int, int]]:
    label_to_position = {label_id: index for index, label_id in enumerate(label_ids)}
    y = np.asarray([label_to_position[row.label_id] for row in rows], dtype=np.int64)
    subjects = [row.subject_id for row in rows]
    split_indices = {
        split: np.asarray(
            [index for index, row in enumerate(rows) if row.split == split],
            dtype=np.int64,
        )
        for split in SPLITS
    }
    return y, subjects, split_indices, label_to_position


def standardize_features(
    features: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, list[float]]]:
    mean = features[train_indices].mean(axis=0)
    std = features[train_indices].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    standardized = (features - mean) / std
    return standardized.astype(np.float32), {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def predict_logits(features: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return features @ weights + bias


def cross_entropy_loss(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    l2: float,
) -> float:
    probabilities = softmax(predict_logits(features, weights, bias))
    selected = probabilities[np.arange(labels.shape[0]), labels]
    loss = -float(np.mean(np.log(np.maximum(selected, 1e-12))))
    if l2:
        loss += 0.5 * l2 * float(np.sum(weights * weights))
    return loss


def local_update(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    lr: float,
    l2: float,
    local_epochs: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    local_weights = weights.copy()
    local_bias = bias.copy()
    class_count = bias.shape[0]
    for _ in range(local_epochs):
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), batch_size):
            batch_indices = shuffled[start : start + batch_size]
            batch_features = features[batch_indices]
            batch_labels = labels[batch_indices]
            probabilities = softmax(
                predict_logits(batch_features, local_weights, local_bias)
            )
            probabilities[np.arange(batch_labels.shape[0]), batch_labels] -= 1.0
            probabilities /= batch_labels.shape[0]
            grad_weights = batch_features.T @ probabilities
            if l2:
                grad_weights += l2 * local_weights
            grad_bias = probabilities.sum(axis=0)
            local_weights -= lr * grad_weights
            local_bias -= lr * grad_bias
    return local_weights, local_bias


def macro_f1(
    labels: np.ndarray, predictions: np.ndarray, class_indices: list[int]
) -> float:
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
    if not f1_values:
        return float("nan")
    return float(np.mean(f1_values))


def per_user_macro_f1_summary(
    labels: np.ndarray,
    predictions: np.ndarray,
    subjects: list[str],
    eval_subjects: set[str],
    supported_threshold: int,
) -> dict[str, float | int | None]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, subject_id in enumerate(subjects):
        grouped[subject_id].append(index)

    values: list[float] = []
    for subject_id in sorted(eval_subjects):
        subject_indices = grouped.get(subject_id, [])
        if not subject_indices:
            continue
        subject_labels = labels[subject_indices]
        subject_predictions = predictions[subject_indices]
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


def evaluate_split(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: list[str],
    indices: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    eval_subjects: set[str],
    l2: float,
) -> dict[str, object]:
    split_features = features[indices]
    split_labels = labels[indices]
    probabilities = softmax(predict_logits(split_features, weights, bias))
    predictions = np.argmax(probabilities, axis=1)
    loss = cross_entropy_loss(split_features, split_labels, weights, bias, l2)
    class_indices = list(range(bias.shape[0]))
    return {
        "loss": loss,
        "accuracy": float(np.mean(predictions == split_labels)),
        "macro_f1": macro_f1(split_labels, predictions, class_indices),
        "per_user": per_user_macro_f1_summary(
            labels=split_labels,
            predictions=predictions,
            subjects=[subjects[int(index)] for index in indices],
            eval_subjects=eval_subjects,
            supported_threshold=SUPPORTED_TEST_CLASS_WINDOWS,
        ),
    }


def evaluate_all_splits(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: list[str],
    split_indices: dict[str, np.ndarray],
    weights: np.ndarray,
    bias: np.ndarray,
    eval_subjects: set[str],
    l2: float,
) -> dict[str, object]:
    return {
        split: evaluate_split(
            features=features,
            labels=labels,
            subjects=subjects,
            indices=indices,
            weights=weights,
            bias=bias,
            eval_subjects=eval_subjects,
            l2=l2,
        )
        for split, indices in split_indices.items()
    }


def flatten_round_metrics(record: dict[str, object]) -> dict[str, object]:
    flattened: dict[str, object] = {
        "round": record["round"],
        "selected_clients": record["selected_clients"],
        "total_communication_bytes": record["communication"]["total_bytes"],
        "uplink_bytes": record["communication"]["uplink_bytes"],
        "downlink_bytes": record["communication"]["downlink_bytes"],
    }
    metrics = record["metrics"]
    assert isinstance(metrics, dict)
    for split in SPLITS:
        split_metrics = metrics[split]
        assert isinstance(split_metrics, dict)
        flattened[f"{split}_loss"] = split_metrics["loss"]
        flattened[f"{split}_accuracy"] = split_metrics["accuracy"]
        flattened[f"{split}_macro_f1"] = split_metrics["macro_f1"]
        per_user = split_metrics["per_user"]
        assert isinstance(per_user, dict)
        flattened[f"{split}_user_mean_macro_f1"] = per_user["mean_macro_f1"]
        flattened[f"{split}_user_worst10_macro_f1"] = per_user[
            "worst_10pct_macro_f1"
        ]
        flattened[f"{split}_user_n"] = per_user["n_users"]
    return flattened


def write_metrics_csv(path: Path, history: list[dict[str, object]]) -> None:
    rows = [flatten_round_metrics(record) for record in history]
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_fedavg(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: list[str],
    split_indices: dict[str, np.ndarray],
    eval_subjects: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    train_indices_by_client: dict[str, np.ndarray] = {}
    train_index_set = set(split_indices["train"].tolist())
    for subject_id in sorted(set(subjects)):
        indices = np.asarray(
            [
                index
                for index, current_subject in enumerate(subjects)
                if current_subject == subject_id and index in train_index_set
            ],
            dtype=np.int64,
        )
        if len(indices) > 0:
            train_indices_by_client[subject_id] = indices

    clients = sorted(train_indices_by_client)
    if not clients:
        raise RuntimeError("no train clients found")
    selected_count = max(1, int(math.ceil(args.client_fraction * len(clients))))
    selected_count = min(selected_count, len(clients))

    feature_count = features.shape[1]
    class_count = int(labels.max()) + 1
    weights = rng.normal(0.0, 0.01, size=(feature_count, class_count)).astype(
        np.float32
    )
    bias = np.zeros(class_count, dtype=np.float32)
    parameter_count = int(weights.size + bias.size)
    parameter_bytes = parameter_count * 4

    history: list[dict[str, object]] = []
    communication = {
        "parameter_count": parameter_count,
        "bytes_per_model": parameter_bytes,
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "total_bytes": 0,
    }

    def record_metrics(round_index: int, selected_clients: int) -> None:
        metrics = evaluate_all_splits(
            features=features,
            labels=labels,
            subjects=subjects,
            split_indices=split_indices,
            weights=weights,
            bias=bias,
            eval_subjects=eval_subjects,
            l2=args.l2,
        )
        history.append(
            {
                "round": round_index,
                "selected_clients": selected_clients,
                "communication": dict(communication),
                "metrics": metrics,
            }
        )

    record_metrics(0, 0)
    for round_index in range(1, args.rounds + 1):
        selected_clients = rng.choice(clients, size=selected_count, replace=False)
        total_examples = 0
        weighted_weights = np.zeros_like(weights)
        weighted_bias = np.zeros_like(bias)
        for client_id in selected_clients:
            client_indices = train_indices_by_client[str(client_id)]
            local_rng = np.random.default_rng(stable_int(args.seed, round_index, client_id))
            local_weights, local_bias = local_update(
                features=features,
                labels=labels,
                indices=client_indices,
                weights=weights,
                bias=bias,
                lr=args.lr,
                l2=args.l2,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                rng=local_rng,
            )
            example_count = int(len(client_indices))
            total_examples += example_count
            weighted_weights += example_count * local_weights
            weighted_bias += example_count * local_bias

        weights = weighted_weights / total_examples
        bias = weighted_bias / total_examples
        communication["uplink_bytes"] += selected_count * parameter_bytes
        communication["downlink_bytes"] += selected_count * parameter_bytes
        communication["total_bytes"] = (
            communication["uplink_bytes"] + communication["downlink_bytes"]
        )

        if round_index % args.eval_every == 0 or round_index == args.rounds:
            record_metrics(round_index, selected_count)

    final_model = {
        "weights": weights.astype(float).tolist(),
        "bias": bias.astype(float).tolist(),
    }
    return history, final_model


def write_json(path: Path, payload: object) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    start_time = time.time()
    if not (0 < args.client_fraction <= 1.0):
        raise ValueError("--client-fraction must be in (0, 1]")
    if args.rounds < 0:
        raise ValueError("--rounds must be non-negative")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, label_ids, eval_subjects, _subject_summary = read_window_manifest(
        args.manifest_dir, args.cohort
    )
    summary = manifest_summary(rows, eval_subjects, args.cohort)
    write_json(args.output_dir / "manifest_sanity.json", summary)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    labels, subjects, split_indices, _label_to_position = build_arrays(rows, label_ids)
    if args.synthetic_smoke:
        features = make_synthetic_features(rows, label_ids, args.seed)
        feature_source = "synthetic_smoke"
    else:
        features = load_archive_features(
            rows, args.archive, skip_archive_sha256=args.skip_archive_sha256
        )
        feature_source = "kuhar_archive_accelerometer_features"
    features, standardization = standardize_features(features, split_indices["train"])

    config = {
        "cohort": args.cohort,
        "feature_source": feature_source,
        "synthetic_smoke": bool(args.synthetic_smoke),
        "manifest_dir": str(args.manifest_dir),
        "archive": str(args.archive),
        "rounds": args.rounds,
        "client_fraction": args.client_fraction,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "l2": args.l2,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "labels": label_ids,
        "feature_count": int(features.shape[1]),
    }
    write_json(args.output_dir / "run_config.json", config)
    write_json(args.output_dir / "feature_standardization.json", standardization)

    history, final_model = train_fedavg(
        features=features,
        labels=labels,
        subjects=subjects,
        split_indices=split_indices,
        eval_subjects=eval_subjects,
        args=args,
    )
    write_json(args.output_dir / "metrics_history.json", history)
    write_json(args.output_dir / "final_metrics.json", history[-1])
    write_metrics_csv(args.output_dir / "round_metrics.csv", history)
    write_json(args.output_dir / "final_model.json", final_model)

    elapsed = time.time() - start_time
    final = flatten_round_metrics(history[-1])
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "elapsed_seconds": elapsed,
                "final_round": final,
                "synthetic_smoke": bool(args.synthetic_smoke),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
