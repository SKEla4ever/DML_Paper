#!/usr/bin/env python3
"""1D CNN FedAvg baseline for the frozen HHAR V1 phone protocol."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
KUHAR_FEDAVG_SCRIPT = Path(__file__).with_name("train_kuhar_1d_cnn_fedavg.py")
spec = importlib.util.spec_from_file_location("shared_fedavg_base", KUHAR_FEDAVG_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load shared FedAvg code at {KUHAR_FEDAVG_SCRIPT}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

HHAR_SCRIPT_DIR = REPO_ROOT / "hhar_delivery" / "scripts"
sys.path.insert(0, str(HHAR_SCRIPT_DIR))
from hhar_data import (  # noqa: E402
    ACTIVITIES,
    EXPECTED_ACTIVITY_ARCHIVE_SHA256,
    WINDOW_SAMPLES,
    iter_label_runs,
    resample_windows,
    split_continuous_segments,
    validate_activity_archive,
)


DEFAULT_MANIFEST_DIR = Path("hhar_delivery/data/processed/hhar")
DEFAULT_ARCHIVE = Path("hhar_delivery/data/raw/hhar/Activity recognition exp.zip")
DEFAULT_OUTPUT_DIR = Path("outputs/hhar_1d_cnn_fedavg_v1")
DEFAULT_CACHE_DIR = Path("outputs/cache")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class WindowRow:
    window_id: str
    recording_id: str
    user_id: str
    client_id: str
    device_id: str
    label_id: int
    split: str
    window_index: int
    start_creation_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "step", "cosine"),
        default="constant",
    )
    parser.add_argument("--lr-step-rounds", type=int, nargs="*", default=[])
    parser.add_argument("--lr-step-gamma", type=float, default=0.1)
    parser.add_argument("--lr-min", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument(
        "--norm", choices=("batchnorm", "groupnorm", "none"), default="batchnorm"
    )
    parser.add_argument("--groupnorm-groups", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--evaluation-splits",
        choices=SPLITS,
        nargs="+",
        default=list(SPLITS),
        help="Dataset splits for metric evaluation; training always uses train only.",
    )
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def read_window_manifest(
    manifest_dir: Path,
) -> tuple[list[WindowRow], set[str], int]:
    eval_clients = set()
    test_supported_metric_clients = 0
    with (manifest_dir / "hhar_client_summary.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["training_client"] == "True":
                eval_clients.add(row["client_id"])
            if row["primary_metric_client"] == "True":
                test_supported_metric_clients += 1

    rows = []
    with gzip.open(
        manifest_dir / "hhar_window_split_manifest.csv.gz", "rt", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            rows.append(
                WindowRow(
                    window_id=row["window_id"],
                    recording_id=row["recording_id"],
                    user_id=row["user_id"],
                    client_id=row["client_id"],
                    device_id=row["device_id"],
                    label_id=int(row["activity_id"]),
                    split=row["split"],
                    window_index=int(row["window_index_in_segment"]),
                    start_creation_ns=int(row["target_start_creation_time_ns"]),
                )
            )
    observed_labels = sorted({row.label_id for row in rows})
    expected_labels = list(range(len(ACTIVITIES)))
    if observed_labels != expected_labels:
        raise RuntimeError(
            f"HHAR manifest labels are {observed_labels}, expected {expected_labels}"
        )
    return rows, eval_clients, test_supported_metric_clients


def manifest_summary(
    rows: list[WindowRow],
    eval_clients: set[str],
    test_supported_metric_clients: int,
) -> dict[str, object]:
    split_counts = Counter(row.split for row in rows)
    client_splits: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        client_splits[row.client_id][row.split] += 1
    return {
        "dataset": "HHAR",
        "manifest_version": "V1",
        "windows": len(rows),
        "input_shape": [3, WINDOW_SAMPLES],
        "labels": list(ACTIVITIES),
        "clients": len(client_splits),
        "metric_candidate_clients": len(eval_clients),
        "test_supported_metric_clients": test_supported_metric_clients,
        "physical_users": len({row.user_id for row in rows}),
        "devices": len({row.device_id for row in rows}),
        "split_windows": {split: split_counts[split] for split in SPLITS},
        "clients_with_train": sum(
            counts["train"] > 0 for counts in client_splits.values()
        ),
    }


def build_label_arrays(
    rows: list[WindowRow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    labels = np.asarray([row.label_id for row in rows], dtype=np.int64)
    clients = np.asarray([row.client_id for row in rows])
    users = np.asarray([row.user_id for row in rows])
    devices = np.asarray([row.device_id for row in rows])
    split_names = np.asarray([row.split for row in rows])
    split_indices = {
        split: np.where(split_names == split)[0].astype(np.int64) for split in SPLITS
    }
    return labels, clients, users, devices, split_indices


def cache_path(cache_dir: Path) -> Path:
    return cache_dir / "hhar_v1_phone_accel_50hz_windows.npz"


def manifest_fingerprint(rows: list[WindowRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row.window_id}|{row.start_creation_ns}\n".encode("utf-8")
        )
    return digest.hexdigest()


def load_or_build_windows(
    rows: list[WindowRow],
    archive_path: Path,
    cache_file: Path,
    rebuild_cache: bool,
) -> np.ndarray:
    expected_fingerprint = manifest_fingerprint(rows)
    if cache_file.exists() and not rebuild_cache:
        with np.load(cache_file, allow_pickle=False) as cached:
            windows = cached["windows"].astype(np.float32)
            cached_fingerprint = (
                str(cached["manifest_fingerprint"].item())
                if "manifest_fingerprint" in cached
                else ""
            )
        if cached_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "HHAR cache does not match the frozen window manifest; "
                "rerun with --rebuild-cache"
            )
        if windows.shape != (len(rows), 3, WINDOW_SAMPLES):
            raise RuntimeError(
                f"HHAR cache shape {windows.shape} does not match manifest "
                f"{(len(rows), 3, WINDOW_SAMPLES)}"
            )
        if not np.isfinite(windows).all():
            raise RuntimeError("HHAR cache contains non-finite values")
        return windows

    validate_activity_archive(archive_path, test_crc=False)
    rows_by_recording: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        rows_by_recording[row.recording_id].append(index)

    windows = np.empty((len(rows), 3, WINDOW_SAMPLES), dtype=np.float32)
    filled = np.zeros(len(rows), dtype=bool)
    for run in iter_label_runs(archive_path, include_values=True):
        if run.activity == "null":
            continue
        for segment in split_continuous_segments(run):
            indices = rows_by_recording.get(segment.recording_id)
            if not indices:
                continue
            ordered_indices = sorted(indices, key=lambda index: rows[index].window_index)
            starts = [rows[index].start_creation_ns for index in ordered_indices]
            segment_windows = resample_windows(segment, starts)
            for local_index, manifest_index in enumerate(ordered_indices):
                windows[manifest_index] = segment_windows[local_index]
                filled[manifest_index] = True

    missing = np.where(~filled)[0]
    if len(missing):
        example = rows[int(missing[0])].window_id
        raise RuntimeError(
            f"failed to rebuild {len(missing)} HHAR windows; first missing={example}"
        )
    if not np.isfinite(windows).all():
        raise RuntimeError("rebuilt HHAR windows contain non-finite values")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        windows=windows,
        manifest_fingerprint=np.asarray(expected_fingerprint),
    )
    return windows


def group_metric_rows(
    model: object,
    windows: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    devices: np.ndarray,
    split_indices: dict[str, np.ndarray],
    args: argparse.Namespace,
    evaluation_splits: tuple[str, ...],
) -> list[dict[str, object]]:
    device = base.choose_device(args.device)
    x_tensor = base.torch.from_numpy(windows)
    y_tensor = base.torch.from_numpy(labels)
    rows = []
    for split in evaluation_splits:
        indices = split_indices[split]
        predictions, _loss = base.predict_split(
            model, x_tensor, y_tensor, indices, device, args.batch_size
        )
        split_labels = labels[indices]
        split_users = users[indices]
        split_devices = devices[indices]
        for grouping, values in (
            ("device", split_devices),
            ("physical_user", split_users),
        ):
            for group_id in sorted(set(values.tolist())):
                mask = values == group_id
                group_labels = split_labels[mask]
                group_predictions = predictions[mask]
                supported = sorted(set(group_labels.tolist()))
                rows.append(
                    {
                        "split": split,
                        "grouping": grouping,
                        "group_id": group_id,
                        "windows": int(np.sum(mask)),
                        "supported_classes": len(supported),
                        "accuracy": float(
                            np.mean(group_predictions == group_labels)
                        ),
                        "macro_f1": base.macro_f1(
                            group_labels, group_predictions, supported
                        ),
                    }
                )
    return rows


def write_group_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    start_time = time.time()
    if not (0 < args.client_fraction <= 1):
        raise ValueError("--client-fraction must be in (0, 1]")
    if args.rounds < 0:
        raise ValueError("--rounds must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, eval_clients, test_supported_metric_clients = read_window_manifest(
        args.manifest_dir
    )
    summary = manifest_summary(
        rows, eval_clients, test_supported_metric_clients
    )
    base.write_json(args.output_dir / "manifest_sanity.json", summary)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    base.require_torch()
    evaluation_splits = tuple(dict.fromkeys(args.evaluation_splits))
    resolved_device = base.choose_device(args.device)
    labels, clients, users, devices, split_indices = build_label_arrays(rows)
    windows = load_or_build_windows(
        rows=rows,
        archive_path=args.archive,
        cache_file=cache_path(args.cache_dir),
        rebuild_cache=args.rebuild_cache,
    )
    windows, standardization = base.standardize_windows(
        windows, split_indices["train"]
    )
    config = {
        "dataset": "HHAR",
        "manifest_version": "V1",
        "manifest_dir": str(args.manifest_dir),
        "archive": str(args.archive),
        "archive_sha256": EXPECTED_ACTIVITY_ARCHIVE_SHA256,
        "cache_dir": str(args.cache_dir),
        "rounds": args.rounds,
        "client_fraction": args.client_fraction,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "lr_step_rounds": args.lr_step_rounds,
        "lr_step_gamma": args.lr_step_gamma,
        "lr_min": args.lr_min,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "norm": args.norm,
        "groupnorm_groups": args.groupnorm_groups,
        "eval_every": args.eval_every,
        "evaluation_splits": list(evaluation_splits),
        "seed": args.seed,
        "device": args.device,
        "requested_device": args.device,
        "resolved_device": str(resolved_device),
        "labels": list(ACTIVITIES),
        "input_shape": [3, WINDOW_SAMPLES],
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch_version": base.torch.__version__,
        "primary_interpretation": "global and per-device; not independent-user fairness",
    }
    base.write_json(args.output_dir / "run_config.json", config)
    base.write_json(
        args.output_dir / "channel_standardization.json", standardization
    )

    history, model = base.train_fedavg(
        windows=windows,
        y=labels,
        subjects=clients,
        split_indices=split_indices,
        eval_subjects=eval_clients,
        args=args,
    )
    base.write_json(args.output_dir / "metrics_history.json", history)
    base.write_json(args.output_dir / "final_metrics.json", history[-1])
    base.write_metrics_csv(args.output_dir / "round_metrics.csv", history)
    base.torch.save(model.state_dict(), args.output_dir / "final_model.pt")
    write_group_metrics(
        args.output_dir / "group_metrics.csv",
        group_metric_rows(
            model=model,
            windows=windows,
            labels=labels,
            users=users,
            devices=devices,
            split_indices=split_indices,
            args=args,
            evaluation_splits=evaluation_splits,
        ),
    )
    final = base.flatten_round_metrics(history[-1])
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
