#!/usr/bin/env python3
"""Generate the frozen HHAR V1 synchronized-execution split manifests."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from hhar_data import (
    ACTIVITIES,
    ACTIVITY_TO_ID,
    DATASET_DOI,
    DATASET_LICENSE,
    DATASET_URL,
    MAX_INTERPOLATION_GAP_NS,
    TARGET_SAMPLE_RATE_HZ,
    WINDOW_DURATION_NS,
    WINDOW_SAMPLES,
    iter_label_runs,
    split_continuous_segments,
    validate_activity_archive,
)


SPLITS = ("train", "validation", "test")
TARGET_RATIOS = {"train": 0.60, "validation": 0.20, "test": 0.20}
EXPECTED_RAW_ROWS = 13_062_475
EXPECTED_USERS = 9
EXPECTED_DEVICES = 8
EXPECTED_MODELS = 4
EXPECTED_EXECUTION_GROUPS = 130
EXPECTED_AMBIGUOUS_SYNCHRONIZATION_RUNS = 1
EXECUTION_OVERLAP_TOLERANCE_MS = 0
SUPPORTED_CLASS_WINDOWS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def ratio_score(counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(
        (counts[split] / total - TARGET_RATIOS[split]) ** 2
        for split in SPLITS
    )


def scan_source(archive_path: Path) -> tuple[list[dict], list[dict], dict]:
    runs = []
    records = []
    raw_label_rows = Counter()
    users: set[str] = set()
    devices: set[str] = set()
    models: set[str] = set()
    raw_clients: set[str] = set()
    raw_rows = 0
    duplicate_timestamps = 0

    for run in iter_label_runs(archive_path, include_values=False):
        run_rows = len(run.creation_times_ns)
        raw_rows += run_rows
        raw_label_rows[run.activity] += run_rows
        users.add(run.user_id)
        devices.add(run.device_id)
        models.add(run.model)
        raw_clients.add(run.client_id)
        run_row = {
            "raw_run_id": run.run_id,
            "user_id": run.user_id,
            "client_id": run.client_id,
            "model": run.model,
            "device_id": run.device_id,
            "activity": run.activity,
            "activity_id": ACTIVITY_TO_ID.get(run.activity),
            "activity_ordinal": run.activity_ordinal,
            "first_source_line": run.first_source_line,
            "last_source_line": run.last_source_line,
            "raw_rows": run_rows,
            "arrival_start_ms": min(run.arrival_times_ms),
            "arrival_end_ms": max(run.arrival_times_ms),
            "arrival_median_ms": int(statistics.median(run.arrival_times_ms)),
            "content_sha256": run.content_sha256,
            "execution_id": "",
        }
        if run.activity == "null":
            continue
        if run.activity not in ACTIVITY_TO_ID:
            raise RuntimeError(f"unexpected HHAR activity label: {run.activity!r}")
        runs.append(run_row)
        segments = split_continuous_segments(run)
        duplicate_timestamps += segments[0].duplicate_creation_timestamps if segments else 0
        for segment in segments:
            median_interval = segment.median_interval_ns
            window_count = segment.window_count
            records.append(
                {
                    "recording_id": segment.recording_id,
                    "raw_run_id": run.run_id,
                    "execution_id": "",
                    "user_id": run.user_id,
                    "client_id": run.client_id,
                    "model": run.model,
                    "device_id": run.device_id,
                    "activity_id": ACTIVITY_TO_ID[run.activity],
                    "activity": run.activity,
                    "segment_index": segment.segment_index,
                    "source_samples": len(segment.creation_times_ns),
                    "start_creation_time_ns": segment.start_creation_ns,
                    "end_creation_time_ns": segment.end_creation_ns,
                    "duration_seconds": segment.duration_ns / 1_000_000_000,
                    "arrival_start_ms": min(segment.arrival_times_ms),
                    "arrival_end_ms": max(segment.arrival_times_ms),
                    "arrival_median_ms": int(
                        statistics.median(segment.arrival_times_ms)
                    ),
                    "gap_before_ns": segment.gap_before_ns,
                    "median_sample_interval_ns": median_interval,
                    "estimated_sample_rate_hz": (
                        1_000_000_000 / median_interval
                        if median_interval and median_interval > 0
                        else None
                    ),
                    "resampled_3s_windows": window_count,
                    "primary_status": "included" if window_count > 0 else "excluded",
                    "exclusion_reason": (
                        "" if window_count > 0 else "insufficient_continuous_duration"
                    ),
                    "split": "",
                }
            )

    if raw_rows != EXPECTED_RAW_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_RAW_ROWS} phone accelerometer rows, found {raw_rows}"
        )
    if len(users) != EXPECTED_USERS or len(devices) != EXPECTED_DEVICES:
        raise RuntimeError(
            "unexpected HHAR identity counts: "
            f"users={len(users)}, devices={len(devices)}"
        )
    if len(models) != EXPECTED_MODELS:
        raise RuntimeError(f"expected {EXPECTED_MODELS} models, found {len(models)}")
    if set(raw_label_rows) != {*ACTIVITIES, "null"}:
        raise RuntimeError(f"unexpected raw labels: {sorted(raw_label_rows)}")

    summary = {
        "raw_rows": raw_rows,
        "raw_label_rows": dict(sorted(raw_label_rows.items())),
        "physical_users": sorted(users),
        "devices": sorted(devices),
        "models": sorted(models),
        "raw_user_device_pairs": len(raw_clients),
        "non_null_label_runs": len(runs),
        "continuous_segments": len(records),
        "duplicate_creation_timestamps": duplicate_timestamps,
        "malformed_rows": 0,
        "nonfinite_rows": 0,
    }
    return runs, records, summary


def assign_execution_groups(runs: list[dict], records: list[dict]) -> list[dict]:
    by_user_activity: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        by_user_activity[(run["user_id"], run["activity"])].append(run)

    groups = []
    for (user_id, activity), activity_runs in sorted(by_user_activity.items()):
        by_device: dict[str, list[dict]] = defaultdict(list)
        for run in activity_runs:
            by_device[run["device_id"]].append(run)
        run_count_frequency = Counter(len(device_runs) for device_runs in by_device.values())
        canonical_count = min(
            run_count_frequency,
            key=lambda count: (-run_count_frequency[count], -count),
        )
        reference_devices = sorted(
            device_id
            for device_id, device_runs in by_device.items()
            if len(device_runs) == canonical_count
        )
        ordered_by_device = {
            device_id: sorted(
                device_runs,
                key=lambda run: (run["arrival_median_ms"], run["raw_run_id"]),
            )
            for device_id, device_runs in by_device.items()
        }
        canonical_executions = []
        for ordinal in range(1, canonical_count + 1):
            reference_runs = [
                ordered_by_device[device_id][ordinal - 1]
                for device_id in reference_devices
            ]
            execution_id = f"user={user_id}/activity={activity}/execution={ordinal:02d}"
            canonical_executions.append(
                {
                    "execution_id": execution_id,
                    "ordinal": ordinal,
                    "arrival_start_ms": int(
                        statistics.median(
                            run["arrival_start_ms"] for run in reference_runs
                        )
                    ),
                    "arrival_end_ms": int(
                        statistics.median(
                            run["arrival_end_ms"] for run in reference_runs
                        )
                    ),
                    "arrival_median_ms": int(
                        statistics.median(
                            run["arrival_median_ms"] for run in reference_runs
                        )
                    ),
                    "reference_devices": reference_devices,
                }
            )

        assignment_by_run: dict[str, str | None] = {}
        for device_id, device_runs in sorted(ordered_by_device.items()):
            if len(device_runs) == canonical_count:
                for run, canonical in zip(device_runs, canonical_executions):
                    assignment_by_run[run["raw_run_id"]] = canonical["execution_id"]
                continue
            for run in device_runs:
                overlapping = [
                    canonical
                    for canonical in canonical_executions
                    if min(run["arrival_end_ms"], canonical["arrival_end_ms"])
                    - max(run["arrival_start_ms"], canonical["arrival_start_ms"])
                    > EXECUTION_OVERLAP_TOLERANCE_MS
                ]
                if len(overlapping) > 1:
                    assignment_by_run[run["raw_run_id"]] = None
                elif len(overlapping) == 1:
                    assignment_by_run[run["raw_run_id"]] = overlapping[0][
                        "execution_id"
                    ]
                else:
                    nearest = min(
                        canonical_executions,
                        key=lambda canonical: (
                            abs(
                                run["arrival_median_ms"]
                                - canonical["arrival_median_ms"]
                            ),
                            canonical["execution_id"],
                        ),
                    )
                    assignment_by_run[run["raw_run_id"]] = nearest["execution_id"]

        for run in activity_runs:
            run["execution_id"] = assignment_by_run[run["raw_run_id"]] or "ambiguous"
        for record in (
            record
            for record in records
            if record["user_id"] == user_id and record["activity"] == activity
        ):
            assignment = assignment_by_run[record["raw_run_id"]]
            if assignment is None:
                record["execution_id"] = "ambiguous"
                record["primary_status"] = "excluded"
                record["exclusion_reason"] = (
                    "ambiguous_synchronization_across_executions"
                )
            else:
                record["execution_id"] = assignment

        records_by_execution: dict[str, list[dict]] = defaultdict(list)
        for record in (
            record
            for record in records
            if record["user_id"] == user_id
            and record["activity"] == activity
            and record["execution_id"] != "ambiguous"
        ):
            records_by_execution[record["execution_id"]].append(record)

        for canonical in canonical_executions:
            execution_id = canonical["execution_id"]
            component = records_by_execution[execution_id]
            raw_run_ids = {record["raw_run_id"] for record in component}
            runs_in_component = [
                run for run in activity_runs if run["raw_run_id"] in raw_run_ids
            ]
            runs_per_device = Counter(run["device_id"] for run in runs_in_component)
            groups.append(
                {
                    "execution_id": execution_id,
                    "user_id": user_id,
                    "activity_id": ACTIVITY_TO_ID[activity],
                    "activity": activity,
                    "arrival_start_ms": canonical["arrival_start_ms"],
                    "arrival_end_ms": canonical["arrival_end_ms"],
                    "raw_runs": len(raw_run_ids),
                    "raw_rows": sum(record["source_samples"] for record in component),
                    "devices": sorted({record["device_id"] for record in component}),
                    "models": sorted({record["model"] for record in component}),
                    "repeated_device_runs": sum(
                        count - 1 for count in runs_per_device.values() if count > 1
                    ),
                    "split": "",
                    "allocation_lock": "",
                }
            )
    return groups


def aggregate_execution_windows(groups: list[dict], records: list[dict]) -> None:
    records_by_execution: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        records_by_execution[record["execution_id"]].append(record)
    for group in groups:
        group_records = records_by_execution[group["execution_id"]]
        included = [
            record for record in group_records if record["primary_status"] == "included"
        ]
        group["continuous_segments"] = len(group_records)
        group["included_segments"] = len(included)
        group["windows"] = sum(record["resampled_3s_windows"] for record in included)
        group["client_windows"] = dict(
            Counter(
                {
                    client_id: sum(
                        record["resampled_3s_windows"]
                        for record in included
                        if record["client_id"] == client_id
                    )
                    for client_id in {record["client_id"] for record in included}
                }
            )
        )
        group["device_windows"] = dict(
            Counter(
                {
                    device_id: sum(
                        record["resampled_3s_windows"]
                        for record in included
                        if record["device_id"] == device_id
                    )
                    for device_id in {record["device_id"] for record in included}
                }
            )
        )


def allocate_execution_groups(groups: list[dict], seed: int) -> dict[str, object]:
    included = [group for group in groups if group["windows"] > 0]
    dimensions = ("class", "user", "client", "device")
    weights = {"class": 30.0, "user": 8.0, "client": 1.0, "device": 8.0}
    totals: dict[str, Counter] = {dimension: Counter() for dimension in dimensions}
    counts: dict[str, dict[object, Counter]] = {
        dimension: defaultdict(Counter) for dimension in dimensions
    }

    def group_contributions(group: dict) -> dict[str, dict[object, int]]:
        return {
            "class": {group["activity_id"]: group["windows"]},
            "user": {group["user_id"]: group["windows"]},
            "client": group["client_windows"],
            "device": group["device_windows"],
        }

    contributions = {
        group["execution_id"]: group_contributions(group) for group in included
    }
    for group in included:
        for dimension, keyed_values in contributions[group["execution_id"]].items():
            totals[dimension].update(keyed_values)

    def adjust(group: dict, split: str, sign: int) -> None:
        for dimension, keyed_values in contributions[group["execution_id"]].items():
            for key, value in keyed_values.items():
                counts[dimension][key][split] += sign * value

    def objective() -> float:
        return sum(
            weights[dimension]
            * ratio_score(counts[dimension][key], total)
            for dimension in dimensions
            for key, total in totals[dimension].items()
        )

    ordered = sorted(
        included,
        key=lambda group: (
            -group["windows"],
            stable_hash(seed, "initial-order", group["execution_id"]),
        ),
    )
    for group in ordered:
        candidates = []
        for split in sorted(
            SPLITS,
            key=lambda value: stable_hash(
                seed, "initial-split", group["execution_id"], value
            ),
        ):
            adjust(group, split, 1)
            candidates.append((objective(), split))
            adjust(group, split, -1)
        group["split"] = min(candidates)[1]
        adjust(group, group["split"], 1)

    def repair_missing_support(
        dimension: str, key: object, destination: str, lock: str
    ) -> bool:
        if counts[dimension][key][destination] > 0:
            return False
        candidates = []
        for group in included:
            contribution = contributions[group["execution_id"]][dimension].get(key, 0)
            if contribution <= 0 or group["split"] == destination:
                continue
            source = group["split"]
            if counts[dimension][key][source] - contribution <= 0:
                continue
            adjust(group, source, -1)
            adjust(group, destination, 1)
            candidates.append((objective(), group["execution_id"], group))
            adjust(group, destination, -1)
            adjust(group, source, 1)
        if not candidates:
            return False
        group = min(candidates)[2]
        source = group["split"]
        adjust(group, source, -1)
        adjust(group, destination, 1)
        group["split"] = destination
        group["allocation_lock"] = lock
        return True

    repairs = []
    for class_id in sorted(totals["class"]):
        for split in SPLITS:
            if repair_missing_support("class", class_id, split, "class_support_repair"):
                repairs.append(f"class:{class_id}:{split}")
    for user_id in sorted(totals["user"]):
        for split in SPLITS:
            if repair_missing_support("user", user_id, split, "user_support_repair"):
                repairs.append(f"user:{user_id}:{split}")
    for client_id in sorted(totals["client"]):
        if repair_missing_support("client", client_id, "train", "client_train_repair"):
            repairs.append(f"client:{client_id}:train")

    accepted_moves = 0
    changes_by_pass = []
    refinement_order = sorted(
        (group for group in included if not group["allocation_lock"]),
        key=lambda group: (
            -group["windows"],
            stable_hash(seed, "refinement-order", group["execution_id"]),
        ),
    )
    for _pass in range(30):
        changes = 0
        for group in refinement_order:
            source = group["split"]
            base_score = objective()
            best = (base_score, source)
            for destination in sorted(
                SPLITS,
                key=lambda value: stable_hash(
                    seed, "refinement-split", group["execution_id"], value
                ),
            ):
                if destination == source:
                    continue
                class_value = contributions[group["execution_id"]]["class"][
                    group["activity_id"]
                ]
                user_value = contributions[group["execution_id"]]["user"][
                    group["user_id"]
                ]
                if counts["class"][group["activity_id"]][source] - class_value <= 0:
                    continue
                if counts["user"][group["user_id"]][source] - user_value <= 0:
                    continue
                if source == "train" and any(
                    counts["client"][client_id]["train"] - value <= 0
                    for client_id, value in group["client_windows"].items()
                ):
                    continue
                adjust(group, source, -1)
                adjust(group, destination, 1)
                candidate = objective()
                adjust(group, destination, -1)
                adjust(group, source, 1)
                if candidate < best[0] - 1e-12:
                    best = (candidate, destination)
            if best[1] != source:
                adjust(group, source, -1)
                adjust(group, best[1], 1)
                group["split"] = best[1]
                accepted_moves += 1
                changes += 1
        changes_by_pass.append(changes)
        if changes == 0:
            break

    for group in groups:
        if group["windows"] == 0:
            group["split"] = "excluded"

    return {
        "method": "deterministic grouped greedy allocation plus coordinate refinement",
        "group_unit": "synchronized physical-user activity execution",
        "target_ratios": TARGET_RATIOS,
        "objective_weights": weights,
        "support_repairs": repairs,
        "refinement_passes": len(changes_by_pass),
        "changes_by_pass": changes_by_pass,
        "accepted_moves": accepted_moves,
        "final_objective": objective(),
    }


def apply_group_splits(groups: list[dict], records: list[dict]) -> None:
    split_by_execution = {group["execution_id"]: group["split"] for group in groups}
    for record in records:
        if record["primary_status"] == "included":
            record["split"] = split_by_execution[record["execution_id"]]
        else:
            record["split"] = "excluded"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_recording_metadata(path: Path, records: list[dict]) -> None:
    fields = [
        "recording_id",
        "raw_run_id",
        "execution_id",
        "user_id",
        "client_id",
        "model",
        "device_id",
        "activity_id",
        "activity",
        "segment_index",
        "source_samples",
        "start_creation_time_ns",
        "end_creation_time_ns",
        "duration_seconds",
        "arrival_start_ms",
        "arrival_end_ms",
        "arrival_median_ms",
        "gap_before_ns",
        "median_sample_interval_ns",
        "estimated_sample_rate_hz",
        "resampled_3s_windows",
        "primary_status",
        "exclusion_reason",
        "split",
    ]
    write_csv(path, sorted(records, key=lambda row: row["recording_id"]), fields)


def write_execution_manifest(path: Path, groups: list[dict]) -> None:
    fields = [
        "execution_id",
        "user_id",
        "activity_id",
        "activity",
        "arrival_start_ms",
        "arrival_end_ms",
        "raw_runs",
        "raw_rows",
        "device_count",
        "devices",
        "model_count",
        "models",
        "repeated_device_runs",
        "continuous_segments",
        "included_segments",
        "windows",
        "split",
        "allocation_lock",
    ]
    rows = []
    for group in groups:
        row = dict(group)
        row["device_count"] = len(group["devices"])
        row["devices"] = ";".join(group["devices"])
        row["model_count"] = len(group["models"])
        row["models"] = ";".join(group["models"])
        rows.append(row)
    write_csv(path, sorted(rows, key=lambda row: row["execution_id"]), fields)


def write_window_manifest(path: Path, records: list[dict]) -> int:
    fields = [
        "window_id",
        "recording_id",
        "raw_run_id",
        "execution_id",
        "user_id",
        "client_id",
        "model",
        "device_id",
        "activity_id",
        "activity",
        "split",
        "window_index_in_segment",
        "target_start_creation_time_ns",
        "target_end_creation_time_ns_exclusive",
        "target_sample_rate_hz",
        "window_samples",
    ]
    row_count = 0
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz:
            with io.TextIOWrapper(gz, newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fields, lineterminator="\n"
                )
                writer.writeheader()
                for record in sorted(
                    (
                        record
                        for record in records
                        if record["primary_status"] == "included"
                    ),
                    key=lambda row: row["recording_id"],
                ):
                    for window_index in range(record["resampled_3s_windows"]):
                        start = (
                            record["start_creation_time_ns"]
                            + window_index * WINDOW_DURATION_NS
                        )
                        writer.writerow(
                            {
                                "window_id": f"{record['recording_id']}/window={window_index:04d}",
                                "recording_id": record["recording_id"],
                                "raw_run_id": record["raw_run_id"],
                                "execution_id": record["execution_id"],
                                "user_id": record["user_id"],
                                "client_id": record["client_id"],
                                "model": record["model"],
                                "device_id": record["device_id"],
                                "activity_id": record["activity_id"],
                                "activity": record["activity"],
                                "split": record["split"],
                                "window_index_in_segment": window_index,
                                "target_start_creation_time_ns": start,
                                "target_end_creation_time_ns_exclusive": start
                                + WINDOW_DURATION_NS,
                                "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
                                "window_samples": WINDOW_SAMPLES,
                            }
                        )
                        row_count += 1
    return row_count


def summarize_and_write(
    output_dir: Path,
    records: list[dict],
    groups: list[dict],
) -> dict[str, object]:
    included = [record for record in records if record["primary_status"] == "included"]
    windows_by_class_split: dict[int, Counter] = defaultdict(Counter)
    groups_by_class_split: dict[int, Counter] = defaultdict(Counter)
    windows_by_client_split: dict[str, Counter] = defaultdict(Counter)
    classes_by_client_split: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    windows_by_device_split: dict[str, Counter] = defaultdict(Counter)
    clients_by_device: dict[str, set[str]] = defaultdict(set)
    segment_rates_by_device: dict[str, list[float]] = defaultdict(list)
    raw_samples_by_device = Counter()

    for record in records:
        raw_samples_by_device[record["device_id"]] += record["source_samples"]
        if record["estimated_sample_rate_hz"] is not None:
            segment_rates_by_device[record["device_id"]].append(
                record["estimated_sample_rate_hz"]
            )
        if record["primary_status"] != "included":
            continue
        split = record["split"]
        windows = record["resampled_3s_windows"]
        windows_by_class_split[record["activity_id"]][split] += windows
        windows_by_client_split[record["client_id"]][split] += windows
        classes_by_client_split[record["client_id"]][split][
            record["activity_id"]
        ] += windows
        windows_by_device_split[record["device_id"]][split] += windows
        clients_by_device[record["device_id"]].add(record["client_id"])
    for group in groups:
        if group["split"] in SPLITS:
            groups_by_class_split[group["activity_id"]][group["split"]] += 1

    class_rows = []
    for class_id, activity in enumerate(ACTIVITIES):
        counts = windows_by_class_split[class_id]
        row = {
            "activity_id": class_id,
            "activity": activity,
            "total_windows": sum(counts.values()),
        }
        for split in SPLITS:
            row[f"{split}_windows"] = counts[split]
            row[f"{split}_executions"] = groups_by_class_split[class_id][split]
        class_rows.append(row)
    write_csv(
        output_dir / "hhar_class_summary.csv",
        class_rows,
        list(class_rows[0].keys()),
    )

    client_rows = []
    for client_id in sorted(windows_by_client_split):
        user_id, device_id = [part.split("=", 1)[1] for part in client_id.split("|")]
        counts = windows_by_client_split[client_id]
        supported_test_classes = sum(
            value >= SUPPORTED_CLASS_WINDOWS
            for value in classes_by_client_split[client_id]["test"].values()
        )
        row = {
            "client_id": client_id,
            "user_id": user_id,
            "device_id": device_id,
            "total_windows": sum(counts.values()),
            **{f"{split}_windows": counts[split] for split in SPLITS},
            **{
                f"{split}_classes": sum(
                    value > 0
                    for value in classes_by_client_split[client_id][split].values()
                )
                for split in SPLITS
            },
            "supported_test_classes": supported_test_classes,
            "training_client": counts["train"] > 0,
            "primary_metric_client": (
                counts["train"] > 0 and supported_test_classes >= 3
            ),
        }
        client_rows.append(row)
    write_csv(
        output_dir / "hhar_client_summary.csv",
        client_rows,
        list(client_rows[0].keys()),
    )

    device_totals = {
        device_id: sum(counts.values())
        for device_id, counts in windows_by_device_split.items()
    }
    median_device_windows = statistics.median(device_totals.values())
    device_rows = []
    for device_id in sorted(device_totals):
        counts = windows_by_device_split[device_id]
        model = next(
            record["model"] for record in records if record["device_id"] == device_id
        )
        rates = segment_rates_by_device[device_id]
        row = {
            "device_id": device_id,
            "model": model,
            "physical_users": len(
                {client.split("|")[0] for client in clients_by_device[device_id]}
            ),
            "clients": len(clients_by_device[device_id]),
            "raw_non_null_samples": raw_samples_by_device[device_id],
            "median_segment_sample_rate_hz": statistics.median(rates),
            "total_windows": device_totals[device_id],
            **{f"{split}_windows": counts[split] for split in SPLITS},
            "weak_device_condition": device_totals[device_id]
            < 0.25 * median_device_windows,
        }
        device_rows.append(row)
    write_csv(
        output_dir / "hhar_device_summary.csv",
        device_rows,
        list(device_rows[0].keys()),
    )

    split_windows = Counter()
    for record in included:
        split_windows[record["split"]] += record["resampled_3s_windows"]
    total_windows = sum(split_windows.values())
    return {
        "included_segments": len(included),
        "excluded_segments": len(records) - len(included),
        "windows": total_windows,
        "split_windows": {split: split_windows[split] for split in SPLITS},
        "split_ratios": {
            split: split_windows[split] / total_windows for split in SPLITS
        },
        "clients": len(client_rows),
        "training_clients": sum(row["training_client"] for row in client_rows),
        "primary_metric_clients": sum(
            row["primary_metric_client"] for row in client_rows
        ),
        "weak_devices": [
            row["device_id"] for row in device_rows if row["weak_device_condition"]
        ],
        "class_support_all_splits": all(
            all(row[f"{split}_windows"] > 0 for split in SPLITS)
            for row in class_rows
        ),
    }


def artifact_sha256s(output_dir: Path, filenames: list[str]) -> dict[str, str]:
    checksums = {}
    for filename in filenames:
        path = output_dir / filename
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        checksums[filename] = digest.hexdigest()
    return checksums


def write_report(path: Path, audit: dict[str, object]) -> None:
    source = audit["source"]
    scan = audit["scan"]
    realized = audit["realized"]
    split_ratios = realized["split_ratios"]
    execution_summary = audit["execution_group_summary"]
    lines = [
        "# HHAR Manifest Audit V1",
        "",
        "## Frozen identity",
        "",
        f"- Source: UCI HHAR, DOI `{DATASET_DOI}`, license `{DATASET_LICENSE}`.",
        "- Primary modality: smartphone three-axis accelerometer only.",
        "- Client: physical-user/device pair; physical user is retained separately.",
        f"- Source archive SHA-256: `{source['archive_sha256']}`.",
        f"- Raw phone accelerometer rows: {scan['raw_rows']:,}; users: "
        f"{len(scan['physical_users'])}; devices: {len(scan['devices'])}; models: "
        f"{len(scan['models'])}; observed user-device pairs: "
        f"{scan['raw_user_device_pairs']}.",
        "",
        "## Preprocessing protocol",
        "",
        "`null` rows define label boundaries but are excluded from supervised windows. "
        "Within each non-null run, samples are ordered by sensor `Creation_Time`; gaps "
        "greater than 1 second create hard continuous-segment boundaries. Each segment "
        "is linearly resampled to 50 Hz and split into non-overlapping 3-second "
        "(150-sample) windows. No window crosses a label transition or sensor gap.",
        "",
        "For each physical user/activity stratum, the majority device run count and "
        "median arrival intervals define canonical synchronized executions. Complete "
        "devices are matched by temporal ordinal; incomplete devices are matched by "
        "interval overlap. A raw run spanning multiple canonical executions is "
        "excluded as ambiguous. Every retained phone view of an execution is assigned "
        "to one split, preventing synchronized-motion leakage.",
        "",
        "## Realized dataset",
        "",
        f"- Synchronized executions: {audit['execution_groups']}; continuous segments: "
        f"{scan['continuous_segments']:,}; included/excluded segments: "
        f"{realized['included_segments']:,}/{realized['excluded_segments']:,}.",
        f"- Phones represented per synchronized execution: "
        f"{execution_summary['minimum_devices_per_execution']}–"
        f"{execution_summary['maximum_devices_per_execution']}; executions with "
        f"a repeated same-device label run: "
        f"{execution_summary['executions_with_repeated_device_runs']}.",
        f"- Raw runs excluded because their arrival interval ambiguously covered "
        f"multiple canonical executions: "
        f"{execution_summary['ambiguous_synchronization_raw_runs']}.",
        f"- Supervised windows: {realized['windows']:,}; train/validation/test: "
        f"{realized['split_windows']['train']:,}/"
        f"{realized['split_windows']['validation']:,}/"
        f"{realized['split_windows']['test']:,} "
        f"({split_ratios['train']:.2%}/{split_ratios['validation']:.2%}/"
        f"{split_ratios['test']:.2%}).",
        f"- Retained user-device clients: {realized['clients']}; clients with train data: "
        f"{realized['training_clients']}; clients meeting the optional >=3 supported "
        f"test-class metric rule: {realized['primary_metric_clients']}.",
        f"- All six activities have support in every split: "
        f"{realized['class_support_all_splits']}.",
        f"- Weak-device condition identified by <25% of median device windows: "
        f"{', '.join(realized['weak_devices']) or 'none'}.",
        "",
        "## Interpretation boundary",
        "",
        "HHAR V1 is a targeted device feature-shift benchmark. Its user-device pairs "
        "are repeated measurements from only nine physical users, so pair-level "
        "dispersion must not be presented as independent-user fairness evidence. "
        "The weak device remains in the primary full-device cohort and must be named "
        "when interpreting quantity skew or FedBN behavior.",
        "",
        "## Integrity",
        "",
        f"- ZIP CRC: {'PASS' if source['zip_crc_check_passed'] else 'not checked'}; "
        f"malformed rows: {scan['malformed_rows']}; non-finite rows: "
        f"{scan['nonfinite_rows']}.",
        "- Split unit is the synchronized execution; each execution appears in exactly "
        "one split and all included clients retain train data.",
        "- Artifact SHA-256 values are recorded in `hhar_split_audit_v1.json`.",
        "",
        "## Source",
        "",
        f"- {DATASET_URL}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_path = args.report or (
        args.output_dir / "hhar_manifest_audit_v1.md"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_summary = validate_activity_archive(args.archive, test_crc=True)
    runs, records, scan_summary = scan_source(args.archive)
    groups = assign_execution_groups(runs, records)
    ambiguous_runs = {
        record["raw_run_id"]
        for record in records
        if record["exclusion_reason"]
        == "ambiguous_synchronization_across_executions"
    }
    if len(groups) != EXPECTED_EXECUTION_GROUPS:
        raise RuntimeError(
            f"expected {EXPECTED_EXECUTION_GROUPS} canonical executions, "
            f"found {len(groups)}"
        )
    if len(ambiguous_runs) != EXPECTED_AMBIGUOUS_SYNCHRONIZATION_RUNS:
        raise RuntimeError(
            f"expected {EXPECTED_AMBIGUOUS_SYNCHRONIZATION_RUNS} ambiguous raw run, "
            f"found {len(ambiguous_runs)}"
        )
    if any(group["repeated_device_runs"] for group in groups):
        raise RuntimeError("canonical HHAR execution contains repeated device runs")
    aggregate_execution_windows(groups, records)
    allocation = allocate_execution_groups(groups, args.seed)
    apply_group_splits(groups, records)

    write_recording_metadata(args.output_dir / "hhar_recording_metadata.csv", records)
    write_execution_manifest(
        args.output_dir / "hhar_execution_split_manifest.csv", groups
    )
    window_rows = write_window_manifest(
        args.output_dir / "hhar_window_split_manifest.csv.gz", records
    )
    realized = summarize_and_write(args.output_dir, records, groups)
    if window_rows != realized["windows"]:
        raise RuntimeError(
            f"window manifest has {window_rows} rows, expected {realized['windows']}"
        )
    if not realized["class_support_all_splits"]:
        raise RuntimeError("at least one HHAR activity is missing from a split")
    if realized["training_clients"] != realized["clients"]:
        raise RuntimeError("at least one included HHAR client has no train windows")

    artifact_names = [
        "hhar_recording_metadata.csv",
        "hhar_execution_split_manifest.csv",
        "hhar_window_split_manifest.csv.gz",
        "hhar_class_summary.csv",
        "hhar_client_summary.csv",
        "hhar_device_summary.csv",
    ]
    audit = {
        "dataset": "HHAR",
        "manifest_version": "V1",
        "dataset_doi": DATASET_DOI,
        "dataset_url": DATASET_URL,
        "dataset_license": DATASET_LICENSE,
        "seed": args.seed,
        "source": source_summary,
        "protocol": {
            "modality": "phone accelerometer",
            "client_unit": "physical-user/device pair",
            "split_unit": "synchronized physical-user activity execution",
            "target_sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "window_samples": WINDOW_SAMPLES,
            "window_seconds": WINDOW_DURATION_NS / 1_000_000_000,
            "window_overlap": 0,
            "maximum_interpolation_gap_ns": MAX_INTERPOLATION_GAP_NS,
            "execution_overlap_tolerance_ms": EXECUTION_OVERLAP_TOLERANCE_MS,
            "synchronization_method": (
                "majority device run-count canonical executions; ordinal matching "
                "for complete devices; overlap matching for incomplete devices"
            ),
            "activity_label_order": list(ACTIVITIES),
        },
        "scan": scan_summary,
        "execution_groups": len(groups),
        "execution_group_summary": {
            "minimum_devices_per_execution": min(
                len(group["devices"]) for group in groups
            ),
            "maximum_devices_per_execution": max(
                len(group["devices"]) for group in groups
            ),
            "executions_with_repeated_device_runs": sum(
                group["repeated_device_runs"] > 0 for group in groups
            ),
            "ambiguous_synchronization_raw_runs": len(
                {
                    record["raw_run_id"]
                    for record in records
                    if record["exclusion_reason"]
                    == "ambiguous_synchronization_across_executions"
                }
            ),
        },
        "allocation": allocation,
        "realized": realized,
        "artifact_sha256": artifact_sha256s(args.output_dir, artifact_names),
    }
    audit_path = args.output_dir / "hhar_split_audit_v1.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(report_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
