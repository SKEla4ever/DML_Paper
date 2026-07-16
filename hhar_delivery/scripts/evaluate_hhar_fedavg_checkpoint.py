#!/usr/bin/env python3
"""Evaluate one frozen HHAR FedAvg checkpoint without retraining."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER = REPO_ROOT / "algorithms/1d_cnn_fedavg/train_hhar_1d_cnn_fedavg.py"
SPEC = importlib.util.spec_from_file_location("hhar_fedavg_trainer", TRAINER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load HHAR FedAvg trainer at {TRAINER}")
hhar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hhar
SPEC.loader.exec_module(hhar)
base = hhar.base


EXPECTED_CONFIG = {
    "dataset": "HHAR",
    "manifest_version": "V1",
    "rounds": 50,
    "client_fraction": 1.0,
    "local_epochs": 1,
    "batch_size": 64,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0,
    "optimizer": "sgd",
    "norm": "batchnorm",
    "groupnorm_groups": 8,
}
EVALUATION_SPLITS = ("validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--split", choices=EVALUATION_SPLITS, required=True)
    parser.add_argument("--archive", type=Path, default=hhar.DEFAULT_ARCHIVE)
    parser.add_argument("--cache-dir", type=Path, default=hhar.DEFAULT_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="mps"
    )
    parser.add_argument("--test-authorization", type=Path)
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


def source_identity(args: argparse.Namespace) -> dict[str, object]:
    run_config_path = args.source_run_dir / "run_config.json"
    checkpoint = args.source_run_dir / "final_model.pt"
    standardization = args.source_run_dir / "channel_standardization.json"
    for path in (run_config_path, checkpoint, standardization):
        if not path.exists():
            raise FileNotFoundError(f"missing frozen source artifact: {path}")
    config = read_json(run_config_path)
    mismatches = [
        f"run_config.{key}: expected {value!r}, got {config.get(key)!r}"
        for key, value in EXPECTED_CONFIG.items()
        if config.get(key) != value
    ]
    if int(config.get("seed", -1)) != args.model_seed:
        mismatches.append("source model seed does not match requested model seed")
    expected_manifest = (
        Path("outputs/hhar_split_seed_sensitivity_v1")
        / f"split_seed{args.split_seed}"
        / "manifest"
    )
    if Path(str(config.get("manifest_dir"))) != expected_manifest:
        mismatches.append(
            f"source manifest is {config.get('manifest_dir')!r}, "
            f"expected {str(expected_manifest)!r}"
        )
    if config.get("resolved_device") != "mps":
        mismatches.append("frozen training run did not resolve to mps")
    if mismatches:
        raise RuntimeError("invalid frozen checkpoint source:\n- " + "\n- ".join(mismatches))
    return {
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
        "source_run_dir": str(args.source_run_dir),
        "source_run_config": str(run_config_path),
        "source_run_config_sha256": file_sha256(run_config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "source_standardization": str(standardization),
        "source_standardization_sha256": file_sha256(standardization),
        "manifest_dir": str(expected_manifest),
        "source_config": config,
    }


def authorize_test(
    args: argparse.Namespace, identity: dict[str, object]
) -> dict[str, object] | None:
    if args.split != "test":
        return None
    if args.test_authorization is None:
        raise RuntimeError(
            "test evaluation requires the tracked locked-test pre-registration"
        )
    authorization = read_json(args.test_authorization)
    if authorization.get("experiment") != "hhar_fedavg_locked_test_v1":
        raise RuntimeError("authorization is not the HHAR FedAvg locked-test V1 protocol")
    if authorization["fixed_evaluation"]["evaluation_split"] != "test":
        raise RuntimeError("authorization does not permit test evaluation")
    if not authorization["fixed_evaluation"]["evaluation_only"]:
        raise RuntimeError("authorization is not evaluation-only")
    matches = [
        row
        for row in authorization["checkpoint_manifest"]
        if int(row["model_seed"]) == args.model_seed
        and int(row["split_seed"]) == args.split_seed
    ]
    if len(matches) != 1:
        raise RuntimeError("authorization does not contain exactly one matching cell")
    registered = matches[0]
    checks = {
        "source_run_dir": identity["source_run_dir"],
        "checkpoint": identity["checkpoint"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_bytes": identity["checkpoint_bytes"],
    }
    mismatches = [
        f"authorization.{key}: expected {value!r}, got {registered.get(key)!r}"
        for key, value in checks.items()
        if registered.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "checkpoint authorization mismatch:\n- " + "\n- ".join(mismatches)
        )
    return {
        "path": str(args.test_authorization),
        "sha256": file_sha256(args.test_authorization),
        "source_result_commit": authorization["source_result_commit"],
    }


def class_metric_rows(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[list[dict[str, object]], np.ndarray]:
    num_classes = len(hhar.ACTIVITIES)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    rows = []
    for class_id, class_name in enumerate(hhar.ACTIVITIES):
        true_positive = int(confusion[class_id, class_id])
        false_positive = int(confusion[:, class_id].sum() - true_positive)
        false_negative = int(confusion[class_id, :].sum() - true_positive)
        support = int(confusion[class_id, :].sum())
        predicted = int(confusion[:, class_id].sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "support": support,
                "predicted": predicted,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows, confusion


def confusion_rows(confusion: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for class_id, class_name in enumerate(hhar.ACTIVITIES):
        row: dict[str, object] = {
            "true_class_id": class_id,
            "true_class_name": class_name,
            "support": int(confusion[class_id].sum()),
        }
        for predicted_id, predicted_name in enumerate(hhar.ACTIVITIES):
            row[f"predicted_{predicted_id}_{predicted_name}"] = int(
                confusion[class_id, predicted_id]
            )
        rows.append(row)
    return rows


def group_metric_rows(
    labels: np.ndarray,
    predictions: np.ndarray,
    clients: np.ndarray,
    users: np.ndarray,
    devices: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for grouping, values in (
        ("device", devices),
        ("physical_user", users),
        ("user_device_pair", clients),
    ):
        for group_id in sorted(set(values.tolist())):
            mask = values == group_id
            group_labels = labels[mask]
            group_predictions = predictions[mask]
            class_counts = Counter(group_labels.tolist())
            supported = sorted(int(value) for value in class_counts)
            rows.append(
                {
                    "grouping": grouping,
                    "group_id": group_id,
                    "windows": int(mask.sum()),
                    "supported_classes": len(supported),
                    "supported_class_ids": ",".join(map(str, supported)),
                    "accuracy": float(np.mean(group_predictions == group_labels)),
                    "supported_class_macro_f1": base.macro_f1(
                        group_labels, group_predictions, supported
                    ),
                }
            )
    return rows


def group_summary(
    rows: list[dict[str, object]], grouping: str
) -> dict[str, object]:
    values = np.asarray(
        [
            float(row["supported_class_macro_f1"])
            for row in rows
            if row["grouping"] == grouping
        ],
        dtype=np.float64,
    )
    ordered = np.sort(values)
    worst_count = max(1, int(math.ceil(0.10 * len(ordered))))
    return {
        "grouping": grouping,
        "n_groups": int(len(values)),
        "mean_supported_class_macro_f1": float(values.mean()),
        "median_supported_class_macro_f1": float(np.median(values)),
        "population_std_supported_class_macro_f1": float(values.std()),
        "minimum_supported_class_macro_f1": float(ordered[0]),
        "worst_10pct_supported_class_macro_f1": float(
            ordered[:worst_count].mean()
        ),
    }


def write_predictions(
    path: Path,
    manifest_rows: list[object],
    indices: np.ndarray,
    predictions: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as handle:
        fieldnames = [
            "window_id",
            "recording_id",
            "physical_user_id",
            "client_id",
            "device_id",
            "true_class_id",
            "true_class_name",
            "predicted_class_id",
            "predicted_class_name",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for manifest_index, prediction in zip(indices.tolist(), predictions.tolist()):
            row = manifest_rows[manifest_index]
            writer.writerow(
                {
                    "window_id": row.window_id,
                    "recording_id": row.recording_id,
                    "physical_user_id": row.user_id,
                    "client_id": row.client_id,
                    "device_id": row.device_id,
                    "true_class_id": row.label_id,
                    "true_class_name": hhar.ACTIVITIES[row.label_id],
                    "predicted_class_id": prediction,
                    "predicted_class_name": hhar.ACTIVITIES[prediction],
                    "correct": row.label_id == prediction,
                }
            )


def main() -> None:
    args = parse_args()
    start_time = time.time()
    identity = source_identity(args)
    if args.dry_run:
        print(json.dumps(identity, indent=2, sort_keys=True))
        return
    authorization = authorize_test(args, identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_config = identity["source_config"]
    manifest_dir = Path(str(identity["manifest_dir"]))
    manifest_rows, eval_clients, test_supported_clients = hhar.read_window_manifest(
        manifest_dir
    )
    manifest_sanity = hhar.manifest_summary(
        manifest_rows, eval_clients, test_supported_clients
    )
    labels, clients, users, devices, split_indices = hhar.build_label_arrays(
        manifest_rows
    )
    windows = hhar.load_or_build_windows(
        rows=manifest_rows,
        archive_path=args.archive,
        cache_file=hhar.cache_path(args.cache_dir),
        rebuild_cache=False,
    )
    windows, computed_standardization = base.standardize_windows(
        windows, split_indices["train"]
    )
    source_standardization = read_json(
        Path(str(identity["source_standardization"]))
    )
    for key in ("channel_mean", "channel_std"):
        if not np.allclose(
            computed_standardization[key],
            source_standardization[key],
            rtol=0.0,
            atol=1e-7,
        ):
            raise RuntimeError(f"computed {key} differs from frozen training transform")

    device = base.choose_device(args.device)
    model_args = argparse.Namespace(
        norm=source_config["norm"],
        groupnorm_groups=source_config["groupnorm_groups"],
    )
    model = base.build_model(len(hhar.ACTIVITIES), model_args)
    state = base.torch.load(
        Path(str(identity["checkpoint"])), map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    x_tensor = base.torch.from_numpy(windows)
    y_tensor = base.torch.from_numpy(labels)
    indices = split_indices[args.split]
    predictions, loss = base.predict_split(
        model, x_tensor, y_tensor, indices, device, args.batch_size
    )
    split_labels = labels[indices]
    split_clients = clients[indices]
    split_users = users[indices]
    split_devices = devices[indices]
    classes, confusion = class_metric_rows(split_labels, predictions)
    macro_f1 = float(np.mean([float(row["f1"]) for row in classes]))
    reference_macro_f1 = base.macro_f1(
        split_labels, predictions, list(range(len(hhar.ACTIVITIES)))
    )
    if not math.isclose(macro_f1, reference_macro_f1, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("independent class metric Macro-F1 disagrees with trainer")
    groups = group_metric_rows(
        split_labels, predictions, split_clients, split_users, split_devices
    )
    pair_summary = base.per_user_macro_f1_summary(
        labels=split_labels,
        predictions=predictions,
        subjects=split_clients,
        eval_subjects=eval_clients,
        supported_threshold=base.SUPPORTED_TEST_CLASS_WINDOWS,
    )
    global_metrics = {
        "windows": int(len(indices)),
        "loss": float(loss),
        "accuracy": float(np.mean(predictions == split_labels)),
        "macro_f1": macro_f1,
        "correct": int(np.sum(predictions == split_labels)),
        "classes": len(hhar.ACTIVITIES),
    }
    evaluation_config = {
        "evaluation_only": True,
        "training_performed": False,
        "split": args.split,
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
        "source_run_dir": identity["source_run_dir"],
        "checkpoint": identity["checkpoint"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "checkpoint_bytes": identity["checkpoint_bytes"],
        "manifest_dir": identity["manifest_dir"],
        "archive": str(args.archive),
        "cache_dir": str(args.cache_dir),
        "batch_size": args.batch_size,
        "requested_device": args.device,
        "resolved_device": str(device),
        "python_version": sys.version.split()[0],
        "torch_version": base.torch.__version__,
        "test_authorization": authorization,
    }
    metrics = {
        "evaluation_config": evaluation_config,
        "manifest_sanity": manifest_sanity,
        "global_metrics": global_metrics,
        "user_device_pair_summary_existing_protocol": pair_summary,
        "physical_user_summary": group_summary(groups, "physical_user"),
        "device_summary": group_summary(groups, "device"),
        "checkpoint_identity": {
            key: value for key, value in identity.items() if key != "source_config"
        },
        "elapsed_seconds": time.time() - start_time,
    }
    write_json(args.output_dir / "evaluation_config.json", evaluation_config)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "manifest_sanity.json", manifest_sanity)
    write_json(
        args.output_dir / "channel_standardization.json",
        computed_standardization,
    )
    write_csv(args.output_dir / "class_metrics.csv", classes)
    write_csv(args.output_dir / "group_metrics.csv", groups)
    write_csv(args.output_dir / "confusion_matrix.csv", confusion_rows(confusion))
    write_predictions(
        args.output_dir / "predictions.csv.gz",
        manifest_rows,
        indices,
        predictions,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
