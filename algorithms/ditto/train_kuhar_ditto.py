#!/usr/bin/env python3
"""Ditto 1D-CNN baseline for the frozen KU-HAR V1 manifest."""

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


DEFAULT_OUTPUT_DIR = Path("outputs/kuhar_ditto_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Ditto 1D-CNN baseline on KU-HAR frozen V1."
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
    parser.add_argument("--global-local-epochs", type=int, default=2)
    parser.add_argument("--personal-local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument("--ditto-lambda", type=float, default=0.1)
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


def clone_state(state: dict[str, object]) -> dict[str, object]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def parameter_state_from_state_dict(
    state: dict[str, object],
    model,
) -> dict[str, object]:
    return {
        name: state[name].detach().cpu().clone()
        for name, _parameter in model.named_parameters()
    }


def proximal_penalty(model, reference_parameters: dict[str, object], device) -> object:
    penalty = None
    for name, parameter in model.named_parameters():
        reference = reference_parameters[name].to(device)
        value = base.torch.sum((parameter - reference) ** 2)
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        raise RuntimeError("model has no parameters for Ditto proximal penalty")
    return penalty


def train_personal_ditto_model(
    personal_state: dict[str, object],
    global_reference_state: dict[str, object],
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
    model.load_state_dict(personal_state)
    model.train()
    optimizer = base.make_optimizer(model, args)
    criterion = base.nn.CrossEntropyLoss()
    reference_parameters = parameter_state_from_state_dict(global_reference_state, model)
    rng = np.random.default_rng(base.stable_int(args.seed, "ditto", round_index, client_id))
    for _epoch in range(args.personal_local_epochs):
        shuffled = client_indices.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            batch_x = x_tensor[batch_indices].to(device)
            batch_y = y_tensor[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            if args.ditto_lambda:
                loss = loss + 0.5 * args.ditto_lambda * proximal_penalty(
                    model, reference_parameters, device
                )
            loss.backward()
            optimizer.step()
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train_global_ditto_model(
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
    global_args = copy.copy(args)
    global_args.local_epochs = args.global_local_epochs
    return base.train_local_model(
        global_state=global_state,
        x_tensor=x_tensor,
        y_tensor=y_tensor,
        client_indices=client_indices,
        num_classes=num_classes,
        device=device,
        args=global_args,
        round_index=round_index,
        client_id=client_id,
    )


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


def evaluate_split_personal(
    personal_states: dict[str, dict[str, object]],
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
        model.load_state_dict(personal_states[subject_id])
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


def evaluate_all_splits_personal(
    personal_states: dict[str, dict[str, object]],
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
        split: evaluate_split_personal(
            personal_states=personal_states,
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


def train_ditto(
    windows: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
    eval_subjects: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], object, dict[str, dict[str, object]]]:
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
    initial_state = clone_state(global_model.state_dict())
    parameter_bytes = base.state_payload_bytes(global_model)
    parameter_count = int(sum(p.numel() for p in global_model.parameters()))

    all_subjects = sorted(str(subject_id) for subject_id in set(subjects.tolist()))
    personal_states = {
        subject_id: clone_state(initial_state)
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
        "parameter_count": parameter_count,
        "bytes_per_model_state": parameter_bytes,
        "personal_model_state_bytes": parameter_bytes,
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "total_bytes": 0,
    }
    history: list[dict[str, object]] = []

    def record_metrics(round_index: int, selected_clients: int) -> None:
        metrics = evaluate_all_splits_personal(
            personal_states=personal_states,
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
        current_global_state = clone_state(global_model.state_dict())
        global_local_states = []
        example_counts = []
        for client_id in selected_clients:
            client_id = str(client_id)
            indices = train_indices_by_client[client_id]
            global_local_states.append(
                train_global_ditto_model(
                    global_state=current_global_state,
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
            personal_states[client_id] = train_personal_ditto_model(
                personal_state=personal_states[client_id],
                global_reference_state=current_global_state,
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                client_indices=indices,
                num_classes=num_classes,
                device=device,
                args=args,
                round_index=round_index,
                client_id=client_id,
            )
            example_counts.append(int(len(indices)))

        averaged_global_state = base.average_state_dicts(global_local_states, example_counts)
        global_model.load_state_dict(averaged_global_state)
        communication["uplink_bytes"] += selected_count * parameter_bytes
        communication["downlink_bytes"] += selected_count * parameter_bytes
        communication["total_bytes"] = (
            communication["uplink_bytes"] + communication["downlink_bytes"]
        )
        if round_index % args.eval_every == 0 or round_index == args.rounds:
            record_metrics(round_index, selected_count)

    return history, global_model, personal_states


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
    if args.global_local_epochs < 0:
        raise ValueError("--global-local-epochs must be non-negative")
    if args.personal_local_epochs < 0:
        raise ValueError("--personal-local-epochs must be non-negative")
    if args.ditto_lambda < 0:
        raise ValueError("--ditto-lambda must be non-negative")

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
        "algorithm": "ditto",
        "cohort": args.cohort,
        "manifest_dir": str(args.manifest_dir),
        "archive": str(args.archive),
        "cache_dir": str(args.cache_dir),
        "rounds": args.rounds,
        "client_fraction": args.client_fraction,
        "global_local_epochs": args.global_local_epochs,
        "personal_local_epochs": args.personal_local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "ditto_lambda": args.ditto_lambda,
        "norm": args.norm,
        "groupnorm_groups": args.groupnorm_groups,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "device": args.device,
        "labels": label_ids,
        "input_shape": [3, base.WINDOW_SAMPLES],
        "torch_version": base.torch.__version__,
        "communication_accounting": (
            "uplink and downlink include global model state only; "
            "personalized model states remain local"
        ),
        "evaluation": "client-specific full personalized model per subject",
    }
    base.write_json(args.output_dir / "run_config.json", config)
    base.write_json(args.output_dir / "channel_standardization.json", standardization)

    history, global_model, personal_states = train_ditto(
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
            "global_state": global_model.state_dict(),
            "personal_states": personal_states,
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
