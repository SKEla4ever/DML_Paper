#!/usr/bin/env python3
"""FedRep 1D-CNN baseline for the frozen KU-HAR V1 manifest."""

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


DEFAULT_OUTPUT_DIR = Path("outputs/kuhar_fedrep_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a FedRep 1D-CNN baseline on KU-HAR frozen V1."
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
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--representation-steps", type=int, default=1)
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


def head_state_keys(model) -> set[str]:
    linear_modules = [
        name for name, module in model.named_modules() if isinstance(module, base.nn.Linear)
    ]
    if not linear_modules:
        raise RuntimeError("FedRep requires a final Linear classifier head")
    head_prefix = linear_modules[-1]
    keys = {
        key
        for key in model.state_dict()
        if key == head_prefix or key.startswith(f"{head_prefix}.")
    }
    if not keys:
        raise RuntimeError(f"could not find state keys for head prefix {head_prefix}")
    return keys


def state_bytes_for_keys(state: dict[str, object], keys: set[str]) -> int:
    return int(sum(state[key].numel() * state[key].element_size() for key in keys))


def parameter_count_for_keys(model, keys: set[str]) -> int:
    return int(
        sum(parameter.numel() for name, parameter in model.named_parameters() if name in keys)
    )


def clone_state_subset(state: dict[str, object], keys: set[str]) -> dict[str, object]:
    return {key: state[key].detach().cpu().clone() for key in keys}


def merge_representation_and_head(
    representation_state: dict[str, object],
    client_head_state: dict[str, object],
    head_keys: set[str],
) -> dict[str, object]:
    return {
        key: (
            client_head_state[key].detach().cpu().clone()
            if key in head_keys
            else value.detach().cpu().clone()
        )
        for key, value in representation_state.items()
    }


def set_trainable_by_keys(model, trainable_keys: set[str]) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in trainable_keys


def make_optimizer(parameters, args: argparse.Namespace):
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        raise RuntimeError("optimizer received no trainable parameters")
    if args.optimizer == "adam":
        return base.torch.optim.Adam(
            params, lr=args.lr, weight_decay=args.weight_decay
        )
    return base.torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def batch_iterator(
    client_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    epochs: int | None = None,
    max_steps: int | None = None,
):
    steps = 0
    epoch_index = 0
    while True:
        if epochs is not None and epoch_index >= epochs:
            break
        shuffled = client_indices.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), batch_size):
            if max_steps is not None and steps >= max_steps:
                return
            steps += 1
            yield shuffled[start : start + batch_size]
        epoch_index += 1
        if epochs is None and max_steps is None:
            break


def train_local_fedrep_model(
    representation_state: dict[str, object],
    client_head_state: dict[str, object],
    head_keys: set[str],
    representation_keys: set[str],
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
    model.load_state_dict(
        merge_representation_and_head(representation_state, client_head_state, head_keys)
    )
    criterion = base.nn.CrossEntropyLoss()

    if args.head_epochs > 0:
        model.eval()
        set_trainable_by_keys(model, head_keys)
        head_optimizer = make_optimizer(model.parameters(), args)
        head_rng = np.random.default_rng(
            base.stable_int(args.seed, "head", round_index, client_id)
        )
        for batch_indices in batch_iterator(
            client_indices=client_indices,
            batch_size=args.batch_size,
            rng=head_rng,
            epochs=args.head_epochs,
        ):
            batch_x = x_tensor[batch_indices].to(device)
            batch_y = y_tensor[batch_indices].to(device)
            head_optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            head_optimizer.step()

    if args.representation_steps > 0:
        model.train()
        set_trainable_by_keys(model, representation_keys)
        representation_optimizer = make_optimizer(model.parameters(), args)
        representation_rng = np.random.default_rng(
            base.stable_int(args.seed, "representation", round_index, client_id)
        )
        for batch_indices in batch_iterator(
            client_indices=client_indices,
            batch_size=args.batch_size,
            rng=representation_rng,
            max_steps=args.representation_steps,
        ):
            batch_x = x_tensor[batch_indices].to(device)
            batch_y = y_tensor[batch_indices].to(device)
            representation_optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            representation_optimizer.step()

    for parameter in model.parameters():
        parameter.requires_grad = True
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def average_representation_state(
    local_states: list[dict[str, object]],
    example_counts: list[int],
    representation_keys: set[str],
) -> dict[str, object]:
    total_examples = float(sum(example_counts))
    averaged: dict[str, object] = {}
    for key in representation_keys:
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


def evaluate_split_fedrep(
    representation_state: dict[str, object],
    client_head_states: dict[str, dict[str, object]],
    head_keys: set[str],
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
            merge_representation_and_head(
                representation_state=representation_state,
                client_head_state=client_head_states[subject_id],
                head_keys=head_keys,
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


def evaluate_all_splits_fedrep(
    representation_state: dict[str, object],
    client_head_states: dict[str, dict[str, object]],
    head_keys: set[str],
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
        split: evaluate_split_fedrep(
            representation_state=representation_state,
            client_head_states=client_head_states,
            head_keys=head_keys,
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


def train_fedrep(
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
    representation_model = base.build_model(num_classes, args).to(device)
    initial_state = copy.deepcopy(representation_model.state_dict())
    head_keys = head_state_keys(representation_model)
    representation_keys = set(initial_state) - head_keys
    representation_bytes = state_bytes_for_keys(initial_state, representation_keys)
    head_bytes = state_bytes_for_keys(initial_state, head_keys)
    total_parameter_count = int(
        sum(parameter.numel() for parameter in representation_model.parameters())
    )
    representation_parameter_count = parameter_count_for_keys(
        representation_model, representation_keys
    )

    all_subjects = sorted(str(subject_id) for subject_id in set(subjects.tolist()))
    client_head_states = {
        subject_id: clone_state_subset(initial_state, head_keys)
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
        "representation_parameter_count": representation_parameter_count,
        "local_head_state_bytes": head_bytes,
        "bytes_per_model_state": representation_bytes,
        "bytes_per_representation_state": representation_bytes,
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "total_bytes": 0,
    }
    history: list[dict[str, object]] = []

    def record_metrics(round_index: int, selected_clients: int) -> None:
        metrics = evaluate_all_splits_fedrep(
            representation_state=representation_model.state_dict(),
            client_head_states=client_head_states,
            head_keys=head_keys,
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
        current_state = copy.deepcopy(representation_model.state_dict())
        local_states = []
        example_counts = []
        for client_id in selected_clients:
            client_id = str(client_id)
            indices = train_indices_by_client[client_id]
            local_state = train_local_fedrep_model(
                representation_state=current_state,
                client_head_state=client_head_states[client_id],
                head_keys=head_keys,
                representation_keys=representation_keys,
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                client_indices=indices,
                num_classes=num_classes,
                device=device,
                args=args,
                round_index=round_index,
                client_id=client_id,
            )
            client_head_states[client_id] = clone_state_subset(local_state, head_keys)
            local_states.append(local_state)
            example_counts.append(int(len(indices)))

        averaged_representation = average_representation_state(
            local_states=local_states,
            example_counts=example_counts,
            representation_keys=representation_keys,
        )
        new_global_state = {
            key: value.detach().cpu().clone()
            for key, value in current_state.items()
        }
        for key, value in averaged_representation.items():
            new_global_state[key] = value
        representation_model.load_state_dict(new_global_state)
        communication["uplink_bytes"] += selected_count * representation_bytes
        communication["downlink_bytes"] += selected_count * representation_bytes
        communication["total_bytes"] = (
            communication["uplink_bytes"] + communication["downlink_bytes"]
        )
        if round_index % args.eval_every == 0 or round_index == args.rounds:
            record_metrics(round_index, selected_count)

    return history, representation_model, client_head_states, head_keys


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
    if args.head_epochs < 0:
        raise ValueError("--head-epochs must be non-negative")
    if args.representation_steps < 0:
        raise ValueError("--representation-steps must be non-negative")

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
        "algorithm": "fedrep",
        "cohort": args.cohort,
        "manifest_dir": str(args.manifest_dir),
        "archive": str(args.archive),
        "cache_dir": str(args.cache_dir),
        "rounds": args.rounds,
        "client_fraction": args.client_fraction,
        "head_epochs": args.head_epochs,
        "representation_steps": args.representation_steps,
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
        "input_shape": [3, base.WINDOW_SAMPLES],
        "torch_version": base.torch.__version__,
        "communication_accounting": (
            "uplink and downlink include shared representation state only"
        ),
        "evaluation": "client-specific local classifier head per subject",
        "protocol_note": (
            "head update freezes representation in eval mode; representation update "
            "freezes head and trains representation"
        ),
    }
    base.write_json(args.output_dir / "run_config.json", config)
    base.write_json(args.output_dir / "channel_standardization.json", standardization)

    history, model, client_head_states, head_keys = train_fedrep(
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
            "representation_state": model.state_dict(),
            "client_head_states": client_head_states,
            "head_state_keys": sorted(head_keys),
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
