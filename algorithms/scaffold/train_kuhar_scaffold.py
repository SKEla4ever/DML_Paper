#!/usr/bin/env python3
"""SCAFFOLD 1D-CNN baseline for the frozen KU-HAR V1 manifest."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import sys
import time
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


DEFAULT_OUTPUT_DIR = Path("outputs/kuhar_scaffold_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a neural SCAFFOLD 1D-CNN baseline on KU-HAR frozen V1."
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
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--client-fraction", type=float, default=1.0)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--norm",
        choices=("batchnorm", "groupnorm", "none"),
        default="batchnorm",
    )
    parser.add_argument("--groupnorm-groups", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
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


def parameter_state(model) -> dict[str, object]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def zero_control_state(model) -> dict[str, object]:
    return {
        name: base.torch.zeros_like(parameter.detach().cpu())
        for name, parameter in model.named_parameters()
    }


def control_state_bytes(control_state: dict[str, object]) -> int:
    return int(
        sum(tensor.numel() * tensor.element_size() for tensor in control_state.values())
    )


def update_parameter_with_correction(
    model,
    server_control: dict[str, object],
    client_control: dict[str, object],
    lr: float,
    weight_decay: float,
    device,
) -> None:
    with base.torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            corrected_gradient = parameter.grad
            if weight_decay:
                corrected_gradient = corrected_gradient + weight_decay * parameter
            corrected_gradient = (
                corrected_gradient
                - client_control[name].to(device)
                + server_control[name].to(device)
            )
            parameter -= lr * corrected_gradient


def compute_new_client_control(
    old_client_control: dict[str, object],
    server_control: dict[str, object],
    global_parameter_state: dict[str, object],
    local_parameter_state: dict[str, object],
    local_steps: int,
    lr: float,
) -> dict[str, object]:
    if local_steps <= 0:
        raise RuntimeError("SCAFFOLD local update made no optimization steps")
    scale = 1.0 / (local_steps * lr)
    return {
        name: (
            old_client_control[name]
            - server_control[name]
            + scale * (global_parameter_state[name] - local_parameter_state[name])
        )
        for name in old_client_control
    }


def control_delta(
    new_control: dict[str, object], old_control: dict[str, object]
) -> dict[str, object]:
    return {
        name: new_control[name] - old_control[name]
        for name in old_control
    }


def train_local_scaffold_model(
    global_state: dict[str, object],
    server_control: dict[str, object],
    client_control: dict[str, object],
    x_tensor,
    y_tensor,
    client_indices: np.ndarray,
    num_classes: int,
    device,
    args: argparse.Namespace,
    round_index: int,
    client_id: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    model = base.build_model(num_classes, args).to(device)
    model.load_state_dict(global_state)
    model.train()
    criterion = base.nn.CrossEntropyLoss()
    global_parameter_state = {
        name: global_state[name].detach().cpu().clone()
        for name, _parameter in model.named_parameters()
    }
    old_client_control = {
        name: tensor.detach().clone()
        for name, tensor in client_control.items()
    }
    rng = np.random.default_rng(base.stable_int(args.seed, round_index, client_id))
    local_steps = 0
    for _epoch in range(args.local_epochs):
        shuffled = client_indices.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), args.batch_size):
            batch_indices = shuffled[start : start + args.batch_size]
            batch_x = x_tensor[batch_indices].to(device)
            batch_y = y_tensor[batch_indices].to(device)
            model.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            update_parameter_with_correction(
                model=model,
                server_control=server_control,
                client_control=client_control,
                lr=args.lr,
                weight_decay=args.weight_decay,
                device=device,
            )
            local_steps += 1

    local_model_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    local_parameter_state = parameter_state(model)
    new_client_control = compute_new_client_control(
        old_client_control=old_client_control,
        server_control=server_control,
        global_parameter_state=global_parameter_state,
        local_parameter_state=local_parameter_state,
        local_steps=local_steps,
        lr=args.lr,
    )
    delta_control = control_delta(new_client_control, old_client_control)
    return local_model_state, new_client_control, delta_control


def average_control_deltas(
    delta_controls: list[dict[str, object]]
) -> dict[str, object]:
    averaged = {}
    for name in delta_controls[0]:
        accumulator = base.torch.zeros_like(delta_controls[0][name], dtype=base.torch.float32)
        for delta in delta_controls:
            accumulator += delta[name].to(dtype=base.torch.float32)
        averaged[name] = accumulator / len(delta_controls)
    return averaged


def apply_server_control_update(
    server_control: dict[str, object],
    average_delta: dict[str, object],
    selected_count: int,
    total_clients: int,
) -> dict[str, object]:
    sample_fraction = selected_count / total_clients
    return {
        name: server_control[name] + sample_fraction * average_delta[name]
        for name in server_control
    }


def train_scaffold(
    windows: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    split_indices: dict[str, np.ndarray],
    eval_subjects: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], object]:
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
    parameter_bytes = base.state_payload_bytes(global_model)
    parameter_count = int(sum(p.numel() for p in global_model.parameters()))
    server_control = zero_control_state(global_model)
    control_bytes = control_state_bytes(server_control)

    train_indices_set = set(split_indices["train"].tolist())
    train_indices_by_client: dict[str, np.ndarray] = {}
    subject_list = subjects.tolist()
    for subject_id in sorted(set(subject_list)):
        indices = np.asarray(
            [
                index
                for index, current_subject in enumerate(subject_list)
                if current_subject == subject_id and index in train_indices_set
            ],
            dtype=np.int64,
        )
        if len(indices) > 0:
            train_indices_by_client[subject_id] = indices

    clients = sorted(train_indices_by_client)
    client_controls = {client_id: zero_control_state(global_model) for client_id in clients}
    selected_count = max(1, int(math.ceil(args.client_fraction * len(clients))))
    selected_count = min(selected_count, len(clients))
    rng = np.random.default_rng(args.seed)
    communication = {
        "parameter_count": parameter_count,
        "bytes_per_model_state": parameter_bytes,
        "bytes_per_control_variate": control_bytes,
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "total_bytes": 0,
    }
    history: list[dict[str, object]] = []

    def record_metrics(round_index: int, selected_clients: int) -> None:
        metrics = base.evaluate_all_splits(
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
        global_state = copy.deepcopy(global_model.state_dict())
        local_states = []
        example_counts = []
        delta_controls = []
        for client_id in selected_clients:
            client_id = str(client_id)
            indices = train_indices_by_client[client_id]
            local_state, new_client_control, delta_control = train_local_scaffold_model(
                global_state=global_state,
                server_control=server_control,
                client_control=client_controls[client_id],
                x_tensor=x_tensor,
                y_tensor=y_tensor,
                client_indices=indices,
                num_classes=num_classes,
                device=device,
                args=args,
                round_index=round_index,
                client_id=client_id,
            )
            client_controls[client_id] = new_client_control
            local_states.append(local_state)
            delta_controls.append(delta_control)
            example_counts.append(int(len(indices)))

        averaged_state = base.average_state_dicts(local_states, example_counts)
        global_model.load_state_dict(averaged_state)
        average_delta = average_control_deltas(delta_controls)
        server_control = apply_server_control_update(
            server_control=server_control,
            average_delta=average_delta,
            selected_count=selected_count,
            total_clients=len(clients),
        )
        communication["uplink_bytes"] += selected_count * (parameter_bytes + control_bytes)
        communication["downlink_bytes"] += selected_count * (parameter_bytes + control_bytes)
        communication["total_bytes"] = (
            communication["uplink_bytes"] + communication["downlink_bytes"]
        )
        if round_index % args.eval_every == 0 or round_index == args.rounds:
            record_metrics(round_index, selected_count)

    return history, global_model


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
    if args.lr <= 0:
        raise ValueError("--lr must be positive")

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
        "algorithm": "scaffold",
        "cohort": args.cohort,
        "manifest_dir": str(args.manifest_dir),
        "archive": str(args.archive),
        "cache_dir": str(args.cache_dir),
        "rounds": args.rounds,
        "client_fraction": args.client_fraction,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "norm": args.norm,
        "groupnorm_groups": args.groupnorm_groups,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "device": args.device,
        "labels": label_ids,
        "input_shape": [3, base.WINDOW_SAMPLES],
        "torch_version": base.torch.__version__,
        "communication_accounting": (
            "uplink and downlink include model state plus control variate payload"
        ),
    }
    base.write_json(args.output_dir / "run_config.json", config)
    base.write_json(args.output_dir / "channel_standardization.json", standardization)

    history, model = train_scaffold(
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
    base.torch.save(model.state_dict(), args.output_dir / "final_model.pt")

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
