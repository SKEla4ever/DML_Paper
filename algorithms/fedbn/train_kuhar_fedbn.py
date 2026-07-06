#!/usr/bin/env python3
"""FedBN 1D-CNN baseline for the frozen KU-HAR V1 manifest."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


FEDAVG_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "1d_cnn_fedavg"
    / "train_kuhar_1d_cnn_fedavg.py"
)
spec = importlib.util.spec_from_file_location("kuhar_fedavg_base", FEDAVG_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load FedAvg base script at {FEDAVG_SCRIPT}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


DEFAULT_OUTPUT_DIR = Path("outputs/kuhar_fedbn_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a FedBN 1D-CNN baseline on KU-HAR frozen V1."
    )
    parser.add_argument("--manifest-dir", type=Path, default=base.DEFAULT_MANIFEST_DIR)
    parser.add_argument("--archive", type=Path, default=base.DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--cohort",
        choices=("minimum_support", "full_sparse"),
        default="minimum_support",
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument("--eval-every", type=int, default=10)
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


def batchnorm_state_keys(model) -> set[str]:
    batchnorm_prefixes = []
    batchnorm_base = base.nn.modules.batchnorm._BatchNorm
    for name, module in model.named_modules():
        if isinstance(module, batchnorm_base):
            batchnorm_prefixes.append(name)
    state_keys = set(model.state_dict())
    keys = {
        key
        for key in state_keys
        if any(key.startswith(f"{prefix}.") for prefix in batchnorm_prefixes)
    }
    if not keys:
        raise RuntimeError("FedBN requires BatchNorm state, but none was found")
    return keys


def state_bytes_for_keys(state: dict[str, object], keys: set[str]) -> int:
    return int(sum(state[key].numel() * state[key].element_size() for key in keys))


def parameter_count_for_keys(model, keys: set[str]) -> int:
    return int(
        sum(parameter.numel() for name, parameter in model.named_parameters() if name in keys)
    )


def clone_state_subset(state: dict[str, object], keys: set[str]) -> dict[str, object]:
    return {key: state[key].detach().cpu().clone() for key in keys}


def merge_shared_and_bn_state(
    shared_state: dict[str, object],
    client_bn_state: dict[str, object],
    bn_keys: set[str],
) -> dict[str, object]:
    return {
        key: (
            client_bn_state[key].detach().cpu().clone()
            if key in bn_keys
            else value.detach().cpu().clone()
        )
        for key, value in shared_state.items()
    }


def average_shared_state(
    local_states: list[dict[str, object]],
    example_counts: list[int],
    shared_keys: set[str],
) -> dict[str, object]:
    total_examples = float(sum(example_counts))
    averaged: dict[str, object] = {}
    for key in shared_keys:
        first = local_states[0][key]
        if base.torch.is_floating_point(first):
            accumulator = base.torch.zeros_like(first, dtype=base.torch.float32)
            for state, count in zip(local_states, example_counts):
                accumulator += state[key].to(dtype=base.torch.float32) * (
                    count / total_examples
                )
            averaged[key] = accumulator.to(dtype=first.dtype)
        else:
            averaged[key] = first.detach().cpu().clone()
    return averaged


def train_local_fedbn_model(
    shared_state: dict[str, object],
    client_bn_state: dict[str, object],
    bn_keys: set[str],
    x_tensor,
    y_tensor,
    client_indices: np.ndarray,
    num_classes: int,
    device,
    args: argparse.Namespace,
    round_index: int,
    client_id: str,
) -> dict[str, object]:
    model = base.build_model(num_classes, args).to(device)
    model.load_state_dict(merge_shared_and_bn_state(shared_state, client_bn_state, bn_keys))
    model.train()
    optimizer = base.make_optimizer(model, args)
    criterion = base.nn.CrossEntropyLoss()
    rng = np.random.default_rng(base.stable_int(args.seed, round_index, client_id))
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


def predict_subject_indices(
    model,
    x_tensor,
    y_tensor,
    indices: np.ndarray,
    device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    model.eval()
    criterion = base.nn.CrossEntropyLoss(reduction="sum")
    predictions: list[np.ndarray] = []
    total_loss = 0.0
    with base.torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_x = x_tensor[batch_indices].to(device)
            batch_y = y_tensor[batch_indices].to(device)
            logits = model(batch_x)
            total_loss += float(criterion(logits, batch_y).detach().cpu())
            predictions.append(
                base.torch.argmax(logits, dim=1).detach().cpu().numpy()
            )
    return np.concatenate(predictions), total_loss


def evaluate_split_fedbn(
    shared_state: dict[str, object],
    client_bn_states: dict[str, dict[str, object]],
    bn_keys: set[str],
    x_tensor,
    y: np.ndarray,
    y_tensor,
    subjects: np.ndarray,
    indices: np.ndarray,
    eval_subjects: set[str],
    device,
    batch_size: int,
    num_classes: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    offsets_by_subject: dict[str, list[int]] = defaultdict(list)
    indices_by_subject: dict[str, list[int]] = defaultdict(list)
    for offset, index in enumerate(indices.tolist()):
        subject_id = str(subjects[index])
        offsets_by_subject[subject_id].append(offset)
        indices_by_subject[subject_id].append(index)

    predictions = np.empty(len(indices), dtype=np.int64)
    total_loss = 0.0
    model = base.build_model(num_classes, args).to(device)
    for subject_id in sorted(indices_by_subject):
        subject_indices = np.asarray(indices_by_subject[subject_id], dtype=np.int64)
        subject_offsets = np.asarray(offsets_by_subject[subject_id], dtype=np.int64)
        model.load_state_dict(
            merge_shared_and_bn_state(
                shared_state=shared_state,
                client_bn_state=client_bn_states[subject_id],
                bn_keys=bn_keys,
            )
        )
        subject_predictions, subject_loss = predict_subject_indices(
            model=model,
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            indices=subject_indices,
            device=device,
            batch_size=batch_size,
        )
        predictions[subject_offsets] = subject_predictions
        total_loss += subject_loss

    labels = y[indices]
    split_subjects = subjects[indices]
    return {
        "loss": float(total_loss / len(indices)),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": base.macro_f1(labels, predictions, list(range(num_classes))),
        "per_user": base.per_user_macro_f1_summary(
            labels=labels,
            predictions=predictions,
            subjects=split_subjects,
            eval_subjects=eval_subjects,
            supported_threshold=base.SUPPORTED_TEST_CLASS_WINDOWS,
        ),
    }


def evaluate_all_splits_fedbn(
    shared_state: dict[str, object],
    client_bn_states: dict[str, dict[str, object]],
    bn_keys: set[str],
    x_tensor,
    y: np.ndarray,
    y_tensor,
    subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
    eval_subjects: set[str],
    device,
    batch_size: int,
    num_classes: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        split: evaluate_split_fedbn(
            shared_state=shared_state,
            client_bn_states=client_bn_states,
            bn_keys=bn_keys,
            x_tensor=x_tensor,
            y=y,
            y_tensor=y_tensor,
            subjects=subjects,
            indices=split_indices[split],
            eval_subjects=eval_subjects,
            device=device,
            batch_size=batch_size,
            num_classes=num_classes,
            args=args,
        )
        for split in base.SPLITS
    }


def train_fedbn(
    windows: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
    eval_subjects: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], object, dict[str, dict[str, object]], set[str]]:
    base.require_torch()
    device = base.choose_device(args.device)
    base.torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        base.torch.cuda.manual_seed_all(args.seed)

    x_tensor = base.torch.from_numpy(windows)
    y_tensor = base.torch.from_numpy(y)
    num_classes = int(y.max()) + 1
    global_model = base.build_model(num_classes, args).to(device)
    global_state = copy.deepcopy(global_model.state_dict())
    bn_keys = batchnorm_state_keys(global_model)
    shared_keys = set(global_state) - bn_keys
    shared_bytes = state_bytes_for_keys(global_state, shared_keys)
    bn_bytes = state_bytes_for_keys(global_state, bn_keys)
    total_parameter_count = int(sum(parameter.numel() for parameter in global_model.parameters()))
    shared_parameter_count = parameter_count_for_keys(global_model, shared_keys)

    all_subjects = sorted(str(subject_id) for subject_id in set(subjects.tolist()))
    client_bn_states = {
        subject_id: clone_state_subset(global_state, bn_keys)
        for subject_id in all_subjects
    }

    train_indices_set = set(split_indices["train"].tolist())
    train_indices_by_client: dict[str, np.ndarray] = {}
    subject_list = subjects.tolist()
    for subject_id in all_subjects:
        indices = np.asarray(
            [
                index
                for index, current_subject in enumerate(subject_list)
                if str(current_subject) == subject_id and index in train_indices_set
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
        "parameter_count": total_parameter_count,
        "shared_parameter_count": shared_parameter_count,
        "local_batchnorm_state_bytes": bn_bytes,
        "bytes_per_model_state": shared_bytes,
        "bytes_per_shared_state": shared_bytes,
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "total_bytes": 0,
    }
    history: list[dict[str, object]] = []

    def record_metrics(round_index: int, selected_clients: int) -> None:
        metrics = evaluate_all_splits_fedbn(
            shared_state=global_model.state_dict(),
            client_bn_states=client_bn_states,
            bn_keys=bn_keys,
            x_tensor=x_tensor,
            y=y,
            y_tensor=y_tensor,
            subjects=subjects,
            split_indices=split_indices,
            eval_subjects=eval_subjects,
            device=device,
            batch_size=args.batch_size,
            num_classes=num_classes,
            args=args,
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
        current_state = copy.deepcopy(global_model.state_dict())
        local_states = []
        example_counts = []
        for client_id in selected_clients:
            client_id = str(client_id)
            indices = train_indices_by_client[client_id]
            local_state = train_local_fedbn_model(
                shared_state=current_state,
                client_bn_state=client_bn_states[client_id],
                bn_keys=bn_keys,
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                client_indices=indices,
                num_classes=num_classes,
                device=device,
                args=args,
                round_index=round_index,
                client_id=client_id,
            )
            client_bn_states[client_id] = clone_state_subset(local_state, bn_keys)
            local_states.append(local_state)
            example_counts.append(int(len(indices)))

        averaged_shared = average_shared_state(local_states, example_counts, shared_keys)
        new_global_state = {
            key: value.detach().cpu().clone()
            for key, value in current_state.items()
        }
        for key, value in averaged_shared.items():
            new_global_state[key] = value
        global_model.load_state_dict(new_global_state)
        communication["uplink_bytes"] += selected_count * shared_bytes
        communication["downlink_bytes"] += selected_count * shared_bytes
        communication["total_bytes"] = (
            communication["uplink_bytes"] + communication["downlink_bytes"]
        )
        if round_index % args.eval_every == 0 or round_index == args.rounds:
            record_metrics(round_index, selected_count)

    return history, global_model, client_bn_states, bn_keys


def write_metrics_csv(path: Path, history: list[dict[str, object]]) -> None:
    rows = [base.flatten_round_metrics(record) for record in history]
    if not rows:
        return
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

    args.norm = "batchnorm"
    args.groupnorm_groups = 8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, label_ids, eval_subjects = base.read_window_manifest(
        args.manifest_dir, args.cohort
    )
    summary = base.manifest_summary(rows, eval_subjects, args.cohort)
    base.write_json(args.output_dir / "manifest_sanity.json", summary)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    base.require_torch()
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

    config = {
        "algorithm": "fedbn",
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
        "eval_every": args.eval_every,
        "seed": args.seed,
        "device": args.device,
        "labels": label_ids,
        "input_shape": [3, base.WINDOW_SAMPLES],
        "torch_version": base.torch.__version__,
        "communication_accounting": (
            "uplink and downlink include shared non-BatchNorm model state only"
        ),
        "evaluation": "client-specific local BatchNorm state per subject",
    }
    base.write_json(args.output_dir / "run_config.json", config)
    base.write_json(args.output_dir / "channel_standardization.json", standardization)

    history, model, client_bn_states, bn_keys = train_fedbn(
        windows=windows,
        y=labels,
        subjects=subjects,
        split_indices=split_indices,
        eval_subjects=eval_subjects,
        args=args,
    )
    base.write_json(args.output_dir / "metrics_history.json", history)
    base.write_json(args.output_dir / "final_metrics.json", history[-1])
    write_metrics_csv(args.output_dir / "round_metrics.csv", history)
    base.torch.save(
        {
            "shared_state": model.state_dict(),
            "client_bn_states": client_bn_states,
            "batchnorm_state_keys": sorted(bn_keys),
        },
        args.output_dir / "final_model.pt",
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
