#!/usr/bin/env python3

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


DATASET_VERSION = 5
DATASET_DOI = "10.17632/45f952y38r.5"
DATASET_URL = "https://data.mendeley.com/datasets/45f952y38r/5"
DATASET_LICENSE = "CC BY 4.0"
EXPECTED_ARCHIVE_SHA256 = (
    "9fe5d0052f2f1d6711afac42ee4badd968116afa8ba4b8ba591f4fdd771c2ec2"
)
WINDOW_SECONDS = 3
SAMPLE_RATE_HZ = 100
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE_HZ
MIN_WINDOWS_PER_SUPPORTED_RECORDING = 3
MIN_SUPPORTED_ACTIVITIES = 3
MIN_COHORT_ACTIVITIES = 8
MIN_COHORT_SECONDS = 300
SPLITS = ("train", "validation", "test")
TARGET_RATIOS = {"train": 0.60, "validation": 0.20, "test": 0.20}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit KU-HAR V5 and generate a deterministic recording split."
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=20260615, type=int)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional report path; defaults to OUTPUT_DIR/kuhar_recording_split_audit_v1.md.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def ratio_score(counts: Counter, total: int) -> float:
    if total <= 0:
        return 0.0
    return sum(
        (counts[split] / total - TARGET_RATIOS[split]) ** 2
        for split in SPLITS
    )


def sensor_stats(
    times: list[float], values: list[tuple[float, float, float]]
) -> dict:
    nonpadding = [
        index
        for index, (timestamp, axes) in enumerate(zip(times, values))
        if timestamp != 0.0 or any(axis != 0.0 for axis in axes)
    ]
    if not nonpadding:
        return {
            "samples": 0,
            "first_time": None,
            "last_time": None,
            "duration_seconds": 0.0,
            "nonmonotonic_steps": 0,
            "duplicate_time_steps": 0,
            "internal_padding_rows": 0,
            "median_step": None,
            "max_step": None,
        }

    last_index = nonpadding[-1]
    valid_times = times[: last_index + 1]
    valid_values = values[: last_index + 1]
    differences = [
        current - previous
        for previous, current in zip(valid_times, valid_times[1:])
    ]
    positive_steps = [difference for difference in differences if difference > 0]
    return {
        "samples": last_index + 1,
        "first_time": valid_times[0],
        "last_time": valid_times[-1],
        "duration_seconds": (
            valid_times[-1] - valid_times[0] if len(valid_times) > 1 else 0.0
        ),
        "nonmonotonic_steps": sum(step < 0 for step in differences),
        "duplicate_time_steps": sum(step == 0 for step in differences),
        "internal_padding_rows": sum(
            timestamp == 0.0 and not any(axis != 0.0 for axis in axes)
            for timestamp, axes in zip(valid_times, valid_values)
        ),
        "median_step": statistics.median(positive_steps) if positive_steps else None,
        "max_step": max(positive_steps) if positive_steps else None,
    }


def scan_archive(archive_path: Path) -> tuple[list[dict], dict]:
    archive_sha256 = file_sha256(archive_path)
    if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            "archive SHA-256 does not match the frozen KU-HAR V5 source: "
            f"{archive_sha256}"
        )

    records = []
    content_hashes = defaultdict(list)
    malformed_column_counts = Counter()
    nonfinite_rows = 0

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC validation failed at {bad_member}")
        members = sorted(
            name for name in archive.namelist() if name.lower().endswith(".csv")
        )
        for member in members:
            folder, filename = member.split("/", 1)
            activity_id_text, activity = folder.split(".", 1)
            filename_parts = Path(filename).stem.split("_")
            subject_id = filename_parts[0]
            activity_code = filename_parts[1] if len(filename_parts) > 1 else ""
            trial_token = filename_parts[2] if len(filename_parts) > 2 else ""
            filename_suffix = "_".join(filename_parts[3:])

            accel_times = []
            accel_values = []
            gyro_times = []
            gyro_values = []
            content_digest = hashlib.sha256()
            malformed_rows = 0

            with archive.open(member) as handle:
                for raw_line in handle:
                    content_digest.update(raw_line)
                    fields = raw_line.strip().split(b",")
                    if len(fields) != 8:
                        malformed_rows += 1
                        malformed_column_counts[len(fields)] += 1
                        continue
                    try:
                        values = [float(field) for field in fields]
                    except ValueError:
                        malformed_rows += 1
                        continue
                    if not all(math.isfinite(value) for value in values):
                        nonfinite_rows += 1
                    accel_times.append(values[0])
                    accel_values.append(tuple(values[1:4]))
                    gyro_times.append(values[4])
                    gyro_values.append(tuple(values[5:8]))

            accel = sensor_stats(accel_times, accel_values)
            gyro = sensor_stats(gyro_times, gyro_values)
            content_sha256 = content_digest.hexdigest()
            content_hashes[content_sha256].append(member)
            records.append(
                {
                    "recording_id": member,
                    "activity_id": int(activity_id_text),
                    "activity": activity,
                    "subject_id": subject_id,
                    "activity_code": activity_code,
                    "trial_token": trial_token,
                    "filename_suffix": filename_suffix,
                    "row_count": len(accel_times),
                    "malformed_rows": malformed_rows,
                    "content_sha256": content_sha256,
                    **{f"accel_{key}": value for key, value in accel.items()},
                    **{f"gyro_{key}": value for key, value in gyro.items()},
                    "accel_3s_windows": accel["samples"] // WINDOW_SAMPLES,
                    "gyro_3s_windows": gyro["samples"] // WINDOW_SAMPLES,
                }
            )

    duplicate_groups = []
    for group_index, members in enumerate(
        sorted(
            (
                sorted(members)
                for members in content_hashes.values()
                if len(members) > 1
            ),
            key=lambda members: members[0],
        ),
        start=1,
    ):
        duplicate_groups.append(
            {
                "duplicate_group_id": f"duplicate_{group_index:02d}",
                "members": members,
            }
        )

    scan_summary = {
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_path.stat().st_size,
        "zip_crc_check_passed": True,
        "recording_files": len(records),
        "malformed_rows": sum(record["malformed_rows"] for record in records),
        "malformed_column_counts": dict(malformed_column_counts),
        "nonfinite_rows": nonfinite_rows,
        "duplicate_groups": duplicate_groups,
    }
    return records, scan_summary


def apply_quality_policy(records: list[dict], duplicate_groups: list[dict]) -> None:
    by_id = {record["recording_id"]: record for record in records}
    for record in records:
        record.update(
            {
                "duplicate_group_id": "",
                "duplicate_group_scope": "",
                "primary_status": "included",
                "exclusion_reason": "",
                "supplementary_6ch_eligible": record["gyro_samples"] > 0,
                "split": "",
                "allocation_lock": "",
            }
        )
        if record["accel_samples"] == 0:
            record["primary_status"] = "excluded"
            record["exclusion_reason"] = "missing_accelerometer"

    for group in duplicate_groups:
        members = group["members"]
        subjects = {by_id[member]["subject_id"] for member in members}
        scope = "same_subject" if len(subjects) == 1 else "cross_subject"
        for member in members:
            by_id[member]["duplicate_group_id"] = group["duplicate_group_id"]
            by_id[member]["duplicate_group_scope"] = scope
        if scope == "same_subject":
            retained = min(members)
            for member in members:
                if member == retained:
                    continue
                by_id[member]["primary_status"] = "excluded"
                by_id[member][
                    "exclusion_reason"
                ] = "exact_duplicate_same_subject_copy"
        else:
            for member in members:
                by_id[member]["primary_status"] = "excluded"
                by_id[member][
                    "exclusion_reason"
                ] = "ambiguous_cross_subject_exact_duplicate"


def derive_cohorts(records: list[dict]) -> dict:
    included = [
        record
        for record in records
        if record["primary_status"] == "included"
        and record["accel_3s_windows"] > 0
    ]
    by_subject = defaultdict(list)
    by_subject_activity = defaultdict(list)
    for record in included:
        by_subject[record["subject_id"]].append(record)
        by_subject_activity[(record["subject_id"], record["activity_id"])].append(
            record
        )

    feasible_activities = defaultdict(list)
    for (subject_id, activity_id), activity_records in by_subject_activity.items():
        qualifying = [
            record
            for record in activity_records
            if record["accel_3s_windows"]
            >= MIN_WINDOWS_PER_SUPPORTED_RECORDING
        ]
        if len(qualifying) >= 2:
            feasible_activities[subject_id].append(activity_id)

    full_sparse_evaluable = {
        subject_id
        for subject_id, activities in feasible_activities.items()
        if len(activities) >= MIN_SUPPORTED_ACTIVITIES
    }
    minimum_support = set()
    for subject_id, subject_records in by_subject.items():
        activity_count = len({record["activity_id"] for record in subject_records})
        duration_seconds = (
            sum(record["accel_samples"] for record in subject_records)
            / SAMPLE_RATE_HZ
        )
        if (
            subject_id in full_sparse_evaluable
            and activity_count >= MIN_COHORT_ACTIVITIES
            and duration_seconds >= MIN_COHORT_SECONDS
        ):
            minimum_support.add(subject_id)

    for record in records:
        subject_id = record["subject_id"]
        record["split_feasible_activity"] = (
            record["activity_id"] in feasible_activities.get(subject_id, [])
        )
        record["full_sparse_evaluable_client"] = (
            subject_id in full_sparse_evaluable
        )
        record["minimum_support_client"] = subject_id in minimum_support

    return {
        "by_subject": by_subject,
        "by_subject_activity": by_subject_activity,
        "feasible_activities": feasible_activities,
        "full_sparse_evaluable": full_sparse_evaluable,
        "minimum_support": minimum_support,
    }


def select_supported_activities(
    subject_id: str,
    activity_ids: list[int],
    by_subject_activity: dict,
    seed: int,
) -> list[int]:
    ranked = []
    for activity_id in activity_ids:
        qualifying = sorted(
            (
                record
                for record in by_subject_activity[(subject_id, activity_id)]
                if record["accel_3s_windows"]
                >= MIN_WINDOWS_PER_SUPPORTED_RECORDING
            ),
            key=lambda record: (
                -record["accel_3s_windows"],
                stable_hash(seed, "support-recording", record["recording_id"]),
            ),
        )
        window_counts = [record["accel_3s_windows"] for record in qualifying]
        ranked.append(
            (
                -window_counts[1],
                -sum(window_counts),
                -len(window_counts),
                stable_hash(seed, "support-activity", subject_id, activity_id),
                activity_id,
            )
        )
    return [item[-1] for item in sorted(ranked)[:MIN_SUPPORTED_ACTIVITIES]]


def reserve_support(records: list[dict], cohorts: dict, seed: int) -> dict:
    reservations = []
    for subject_id in sorted(cohorts["full_sparse_evaluable"], key=int):
        selected_activities = select_supported_activities(
            subject_id,
            cohorts["feasible_activities"][subject_id],
            cohorts["by_subject_activity"],
            seed,
        )
        for activity_id in selected_activities:
            qualifying = sorted(
                (
                    record
                    for record in cohorts["by_subject_activity"][
                        (subject_id, activity_id)
                    ]
                    if record["accel_3s_windows"]
                    >= MIN_WINDOWS_PER_SUPPORTED_RECORDING
                ),
                key=lambda record: (
                    -record["accel_3s_windows"],
                    stable_hash(
                        seed, "reserve-recording", record["recording_id"]
                    ),
                ),
            )
            train_record, test_record = qualifying[:2]
            train_record["split"] = "train"
            train_record["allocation_lock"] = "supported_train_reservation"
            test_record["split"] = "test"
            test_record["allocation_lock"] = "supported_test_reservation"
            reservations.append(
                {
                    "subject_id": subject_id,
                    "activity_id": activity_id,
                    "train_recording_id": train_record["recording_id"],
                    "test_recording_id": test_record["recording_id"],
                }
            )

    for subject_id, subject_records in sorted(
        cohorts["by_subject"].items(), key=lambda item: int(item[0])
    ):
        if any(record["split"] == "train" for record in subject_records):
            continue
        anchor = min(
            subject_records,
            key=lambda record: (
                -record["accel_3s_windows"],
                stable_hash(seed, "train-anchor", record["recording_id"]),
            ),
        )
        anchor["split"] = "train"
        anchor["allocation_lock"] = "client_train_anchor"

    return {
        "support_reservations": reservations,
        "support_reservation_count": len(reservations),
    }


def allocate_recordings(records: list[dict], seed: int) -> dict:
    included = [
        record for record in records if record["primary_status"] == "included"
    ]
    class_counts = defaultdict(Counter)
    minimum_class_counts = defaultdict(Counter)
    client_counts = defaultdict(Counter)
    client_class_counts = defaultdict(Counter)
    class_totals = Counter()
    minimum_class_totals = Counter()
    client_totals = Counter()
    client_class_totals = Counter()

    for record in included:
        activity_id = record["activity_id"]
        subject_id = record["subject_id"]
        windows = record["accel_3s_windows"]
        class_totals[activity_id] += windows
        if record["minimum_support_client"]:
            minimum_class_totals[activity_id] += windows
        client_totals[subject_id] += windows
        client_class_totals[(subject_id, activity_id)] += windows
        if record["split"]:
            split = record["split"]
            class_counts[activity_id][split] += windows
            if record["minimum_support_client"]:
                minimum_class_counts[activity_id][split] += windows
            client_counts[subject_id][split] += windows
            client_class_counts[(subject_id, activity_id)][split] += windows

    global_class_weight = 2.0 * len(client_totals)
    minimum_class_weight = 2.0 * len(
        {
            record["subject_id"]
            for record in included
            if record["minimum_support_client"]
        }
    )
    client_total_weight = 1.0
    client_class_weight = 0.5

    def affected_score(subject_id: str, activity_id: int) -> float:
        score = (
            global_class_weight
            * ratio_score(class_counts[activity_id], class_totals[activity_id])
            + client_total_weight
            * ratio_score(client_counts[subject_id], client_totals[subject_id])
            + client_class_weight
            * ratio_score(
                client_class_counts[(subject_id, activity_id)],
                client_class_totals[(subject_id, activity_id)],
            )
        )
        if minimum_class_totals[activity_id] > 0:
            score += minimum_class_weight * ratio_score(
                minimum_class_counts[activity_id],
                minimum_class_totals[activity_id],
            )
        return score

    unlocked = sorted(
        (record for record in included if not record["split"]),
        key=lambda record: (
            -record["accel_3s_windows"],
            stable_hash(seed, "initial-order", record["recording_id"]),
        ),
    )
    for record in unlocked:
        subject_id = record["subject_id"]
        activity_id = record["activity_id"]
        windows = record["accel_3s_windows"]
        candidates = []
        for split in sorted(
            SPLITS,
            key=lambda candidate: stable_hash(
                seed, "initial-split", record["recording_id"], candidate
            ),
        ):
            class_counts[activity_id][split] += windows
            if record["minimum_support_client"]:
                minimum_class_counts[activity_id][split] += windows
            client_counts[subject_id][split] += windows
            client_class_counts[(subject_id, activity_id)][split] += windows
            candidates.append((affected_score(subject_id, activity_id), split))
            class_counts[activity_id][split] -= windows
            if record["minimum_support_client"]:
                minimum_class_counts[activity_id][split] -= windows
            client_counts[subject_id][split] -= windows
            client_class_counts[(subject_id, activity_id)][split] -= windows
        split = min(candidates)[1]
        record["split"] = split
        class_counts[activity_id][split] += windows
        if record["minimum_support_client"]:
            minimum_class_counts[activity_id][split] += windows
        client_counts[subject_id][split] += windows
        client_class_counts[(subject_id, activity_id)][split] += windows

    refinement_order = sorted(
        (record for record in included if not record["allocation_lock"]),
        key=lambda record: (
            -record["accel_3s_windows"],
            stable_hash(seed, "refinement-order", record["recording_id"]),
        ),
    )
    total_moves = 0
    changes_by_pass = []
    for _ in range(30):
        changes = 0
        for record in refinement_order:
            subject_id = record["subject_id"]
            activity_id = record["activity_id"]
            windows = record["accel_3s_windows"]
            source = record["split"]
            best_score = affected_score(subject_id, activity_id)
            best_split = source
            for destination in sorted(
                SPLITS,
                key=lambda candidate: stable_hash(
                    seed,
                    "refinement-split",
                    record["recording_id"],
                    candidate,
                ),
            ):
                if destination == source:
                    continue
                if (
                    source == "train"
                    and client_counts[subject_id]["train"] - windows <= 0
                ):
                    continue
                class_counts[activity_id][source] -= windows
                class_counts[activity_id][destination] += windows
                if record["minimum_support_client"]:
                    minimum_class_counts[activity_id][source] -= windows
                    minimum_class_counts[activity_id][destination] += windows
                client_counts[subject_id][source] -= windows
                client_counts[subject_id][destination] += windows
                client_class_counts[(subject_id, activity_id)][source] -= windows
                client_class_counts[(subject_id, activity_id)][
                    destination
                ] += windows
                candidate_score = affected_score(subject_id, activity_id)
                class_counts[activity_id][destination] -= windows
                class_counts[activity_id][source] += windows
                if record["minimum_support_client"]:
                    minimum_class_counts[activity_id][destination] -= windows
                    minimum_class_counts[activity_id][source] += windows
                client_counts[subject_id][destination] -= windows
                client_counts[subject_id][source] += windows
                client_class_counts[(subject_id, activity_id)][
                    destination
                ] -= windows
                client_class_counts[(subject_id, activity_id)][source] += windows
                if candidate_score < best_score - 1e-12:
                    best_score = candidate_score
                    best_split = destination

            if best_split == source:
                continue
            class_counts[activity_id][source] -= windows
            class_counts[activity_id][best_split] += windows
            if record["minimum_support_client"]:
                minimum_class_counts[activity_id][source] -= windows
                minimum_class_counts[activity_id][best_split] += windows
            client_counts[subject_id][source] -= windows
            client_counts[subject_id][best_split] += windows
            client_class_counts[(subject_id, activity_id)][source] -= windows
            client_class_counts[(subject_id, activity_id)][best_split] += windows
            record["split"] = best_split
            changes += 1
            total_moves += 1
        changes_by_pass.append(changes)
        if changes == 0:
            break

    for record in records:
        if record["primary_status"] == "excluded":
            record["split"] = "excluded"

    return {
        "method": (
            "support reservations plus deterministic joint greedy allocation "
            "and coordinate refinement"
        ),
        "objective": (
            "2*N_clients*full_class_ratio_error + "
            "2*N_minimum_clients*minimum_cohort_class_ratio_error + "
            "client_total_ratio_error + 0.5*client_class_ratio_error"
        ),
        "global_class_weight": global_class_weight,
        "minimum_class_weight": minimum_class_weight,
        "client_total_weight": client_total_weight,
        "client_class_weight": client_class_weight,
        "refinement_passes": len(changes_by_pass),
        "changes_by_pass": changes_by_pass,
        "accepted_moves": total_moves,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_recording_metadata(path: Path, records: list[dict]) -> None:
    fields = [
        "recording_id",
        "subject_id",
        "activity_id",
        "activity",
        "activity_code",
        "trial_token",
        "filename_suffix",
        "row_count",
        "malformed_rows",
        "content_sha256",
        "duplicate_group_id",
        "duplicate_group_scope",
        "primary_status",
        "exclusion_reason",
        "supplementary_6ch_eligible",
        "accel_samples",
        "accel_duration_seconds",
        "accel_nonmonotonic_steps",
        "accel_duplicate_time_steps",
        "accel_internal_padding_rows",
        "accel_median_step",
        "accel_max_step",
        "gyro_samples",
        "gyro_duration_seconds",
        "gyro_nonmonotonic_steps",
        "gyro_duplicate_time_steps",
        "gyro_internal_padding_rows",
        "gyro_median_step",
        "gyro_max_step",
        "accel_3s_windows",
        "gyro_3s_windows",
        "split_feasible_activity",
        "full_sparse_evaluable_client",
        "minimum_support_client",
        "split",
        "allocation_lock",
    ]
    write_csv(
        path,
        sorted(
            records,
            key=lambda record: (
                int(record["subject_id"]),
                record["activity_id"],
                record["recording_id"],
            ),
        ),
        fields,
    )


def write_recording_manifest(path: Path, records: list[dict]) -> None:
    fields = [
        "recording_id",
        "subject_id",
        "activity_id",
        "activity",
        "accel_samples",
        "accel_3s_windows",
        "tail_samples_dropped",
        "primary_status",
        "exclusion_reason",
        "split",
        "allocation_lock",
        "split_feasible_activity",
        "full_sparse_evaluable_client",
        "minimum_support_client",
        "content_sha256",
    ]
    rows = []
    for record in records:
        row = dict(record)
        row["tail_samples_dropped"] = record["accel_samples"] % WINDOW_SAMPLES
        rows.append(row)
    write_csv(
        path,
        sorted(
            rows,
            key=lambda record: (
                int(record["subject_id"]),
                record["activity_id"],
                record["recording_id"],
            ),
        ),
        fields,
    )


def write_window_manifest(path: Path, records: list[dict]) -> int:
    fields = [
        "recording_id",
        "subject_id",
        "activity_id",
        "activity",
        "split",
        "window_index_in_recording",
        "start_sample_offset",
        "end_sample_offset_exclusive",
        "start_seconds",
        "end_seconds_exclusive",
    ]
    row_count = 0
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, mtime=0
        ) as compressed_handle:
            with io.TextIOWrapper(
                compressed_handle, newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for record in sorted(
                    (
                        record
                        for record in records
                        if record["primary_status"] == "included"
                    ),
                    key=lambda record: (
                        int(record["subject_id"]),
                        record["activity_id"],
                        record["recording_id"],
                    ),
                ):
                    for window_index in range(record["accel_3s_windows"]):
                        start_sample = window_index * WINDOW_SAMPLES
                        end_sample = start_sample + WINDOW_SAMPLES
                        writer.writerow(
                            {
                                "recording_id": record["recording_id"],
                                "subject_id": record["subject_id"],
                                "activity_id": record["activity_id"],
                                "activity": record["activity"],
                                "split": record["split"],
                                "window_index_in_recording": window_index,
                                "start_sample_offset": start_sample,
                                "end_sample_offset_exclusive": end_sample,
                                "start_seconds": f"{start_sample / SAMPLE_RATE_HZ:.2f}",
                                "end_seconds_exclusive": (
                                    f"{end_sample / SAMPLE_RATE_HZ:.2f}"
                                ),
                            }
                        )
                        row_count += 1
    return row_count


def summarize(records: list[dict], cohorts: dict) -> tuple[list[dict], list[dict]]:
    included = [
        record for record in records if record["primary_status"] == "included"
    ]
    by_subject = defaultdict(list)
    by_activity = defaultdict(list)
    for record in included:
        by_subject[record["subject_id"]].append(record)
        by_activity[(record["activity_id"], record["activity"])].append(record)

    subject_rows = []
    for subject_id in sorted(
        {record["subject_id"] for record in records}, key=int
    ):
        raw_records = [
            record for record in records if record["subject_id"] == subject_id
        ]
        subject_records = by_subject[subject_id]
        split_windows = Counter()
        split_recordings = Counter()
        class_split_windows = defaultdict(Counter)
        for record in subject_records:
            split_windows[record["split"]] += record["accel_3s_windows"]
            split_recordings[record["split"]] += 1
            class_split_windows[record["activity_id"]][
                record["split"]
            ] += record["accel_3s_windows"]

        supported_test = {
            activity_id
            for activity_id, counts in class_split_windows.items()
            if counts["test"] >= MIN_WINDOWS_PER_SUPPORTED_RECORDING
        }
        seen_supported = {
            activity_id
            for activity_id in supported_test
            if class_split_windows[activity_id]["train"]
            >= MIN_WINDOWS_PER_SUPPORTED_RECORDING
        }
        unseen_supported = supported_test - seen_supported
        subject_rows.append(
            {
                "subject_id": subject_id,
                "raw_recordings": len(raw_records),
                "retained_recordings": len(subject_records),
                "observed_activities": len(
                    {record["activity_id"] for record in subject_records}
                ),
                "accel_duration_seconds": (
                    sum(record["accel_samples"] for record in subject_records)
                    / SAMPLE_RATE_HZ
                ),
                "split_feasible_activities": len(
                    cohorts["feasible_activities"].get(subject_id, [])
                ),
                "full_sparse_evaluable_client": (
                    subject_id in cohorts["full_sparse_evaluable"]
                ),
                "minimum_support_client": (
                    subject_id in cohorts["minimum_support"]
                ),
                "train_recordings": split_recordings["train"],
                "validation_recordings": split_recordings["validation"],
                "test_recordings": split_recordings["test"],
                "train_windows": split_windows["train"],
                "validation_windows": split_windows["validation"],
                "test_windows": split_windows["test"],
                "supported_test_classes": len(supported_test),
                "seen_supported_test_classes": len(seen_supported),
                "locally_unseen_supported_test_classes": len(unseen_supported),
                "realized_evaluable_client": (
                    len(supported_test) >= MIN_SUPPORTED_ACTIVITIES
                ),
            }
        )

    class_rows = []
    for (activity_id, activity), activity_records in sorted(by_activity.items()):
        split_windows = Counter()
        split_recordings = Counter()
        for record in activity_records:
            split_windows[record["split"]] += record["accel_3s_windows"]
            split_recordings[record["split"]] += 1
        total_windows = sum(split_windows.values())
        class_rows.append(
            {
                "activity_id": activity_id,
                "activity": activity,
                "retained_recordings": len(activity_records),
                "total_windows": total_windows,
                **{
                    f"{split}_recordings": split_recordings[split]
                    for split in SPLITS
                },
                **{
                    f"{split}_windows": split_windows[split] for split in SPLITS
                },
                **{
                    f"{split}_ratio": (
                        split_windows[split] / total_windows
                        if total_windows
                        else 0.0
                    )
                    for split in SPLITS
                },
                "max_absolute_ratio_deviation": max(
                    abs(
                        split_windows[split] / total_windows
                        - TARGET_RATIOS[split]
                    )
                    for split in SPLITS
                ),
            }
        )
    return subject_rows, class_rows


def summarize_class_subset(records: list[dict], predicate) -> list[dict]:
    by_activity = defaultdict(list)
    for record in records:
        if record["primary_status"] == "included" and predicate(record):
            by_activity[(record["activity_id"], record["activity"])].append(record)

    rows = []
    for (activity_id, activity), activity_records in sorted(by_activity.items()):
        split_windows = Counter()
        split_recordings = Counter()
        for record in activity_records:
            split_windows[record["split"]] += record["accel_3s_windows"]
            split_recordings[record["split"]] += 1
        total_windows = sum(split_windows.values())
        rows.append(
            {
                "activity_id": activity_id,
                "activity": activity,
                "retained_recordings": len(activity_records),
                "total_windows": total_windows,
                **{
                    f"{split}_recordings": split_recordings[split]
                    for split in SPLITS
                },
                **{
                    f"{split}_windows": split_windows[split] for split in SPLITS
                },
                **{
                    f"{split}_ratio": split_windows[split] / total_windows
                    for split in SPLITS
                },
                "max_absolute_ratio_deviation": max(
                    abs(
                        split_windows[split] / total_windows
                        - TARGET_RATIOS[split]
                    )
                    for split in SPLITS
                ),
            }
        )
    return rows


def build_audit(
    records: list[dict],
    scan_summary: dict,
    cohorts: dict,
    reservation_metadata: dict,
    allocation_metadata: dict,
    subject_rows: list[dict],
    class_rows: list[dict],
    minimum_class_rows: list[dict],
    window_manifest_rows: int,
    seed: int,
    artifacts: dict[str, Path],
) -> dict:
    included = [
        record for record in records if record["primary_status"] == "included"
    ]
    exclusions = Counter(
        record["exclusion_reason"]
        for record in records
        if record["primary_status"] == "excluded"
    )
    split_windows = Counter()
    split_recordings = Counter()
    for record in included:
        split_windows[record["split"]] += record["accel_3s_windows"]
        split_recordings[record["split"]] += 1
    total_windows = sum(split_windows.values())
    minimum_records = [
        record for record in included if record["minimum_support_client"]
    ]
    minimum_split_windows = Counter()
    for record in minimum_records:
        minimum_split_windows[record["split"]] += record["accel_3s_windows"]
    minimum_total_windows = sum(minimum_split_windows.values())

    realized_evaluable = {
        row["subject_id"]
        for row in subject_rows
        if row["realized_evaluable_client"]
    }
    minimum_realized = realized_evaluable & cohorts["minimum_support"]
    evaluable_rows = [
        row
        for row in subject_rows
        if row["subject_id"] in cohorts["full_sparse_evaluable"]
    ]
    minimum_rows = [
        row
        for row in subject_rows
        if row["subject_id"] in cohorts["minimum_support"]
    ]
    supported_pairs = sum(
        row["supported_test_classes"] for row in evaluable_rows
    )
    seen_pairs = sum(
        row["seen_supported_test_classes"] for row in evaluable_rows
    )
    unseen_pairs = sum(
        row["locally_unseen_supported_test_classes"]
        for row in evaluable_rows
    )
    mismatch_records = [
        record
        for record in records
        if record["accel_samples"] != record["gyro_samples"]
    ]

    artifact_metadata = {
        name: {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for name, path in artifacts.items()
    }
    integrity_checks = {
        "all_retained_recordings_assigned_once": all(
            record["split"] in SPLITS for record in included
        ),
        "all_excluded_recordings_marked_excluded": all(
            record["split"] == "excluded"
            for record in records
            if record["primary_status"] == "excluded"
        ),
        "window_manifest_matches_retained_window_count": (
            window_manifest_rows == total_windows
        ),
        "every_usable_client_has_train_data": all(
            row["train_windows"] > 0
            for row in subject_rows
            if row["retained_recordings"] > 0
        ),
        "all_frozen_evaluable_clients_realized": (
            cohorts["full_sparse_evaluable"] <= realized_evaluable
        ),
        "all_minimum_support_clients_realized": (
            cohorts["minimum_support"] <= minimum_realized
        ),
        "cross_subject_exact_duplicates_excluded": all(
            record["primary_status"] == "excluded"
            for record in records
            if record["duplicate_group_scope"] == "cross_subject"
        ),
        "all_classes_present_in_all_splits": all(
            all(row[f"{split}_windows"] > 0 for split in SPLITS)
            for row in class_rows
        ),
        "all_minimum_cohort_classes_present_in_all_splits": all(
            all(row[f"{split}_windows"] > 0 for split in SPLITS)
            for row in minimum_class_rows
        ),
    }
    if not all(integrity_checks.values()):
        raise RuntimeError(
            "manifest integrity check failed: "
            + json.dumps(integrity_checks, sort_keys=True)
        )

    return {
        "audit_version": "KU-HAR recording split audit V1",
        "audit_date": "2026-06-15",
        "source": {
            "dataset": "KU-HAR",
            "version": DATASET_VERSION,
            "doi": DATASET_DOI,
            "url": DATASET_URL,
            "license": DATASET_LICENSE,
            "archive_filename": "2.Trimmed_interpolated_data.zip",
            **scan_summary,
        },
        "primary_protocol": {
            "modality": "three-axis accelerometer",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "window_seconds": WINDOW_SECONDS,
            "window_samples": WINDOW_SAMPLES,
            "window_overlap": 0.0,
            "split_unit": "complete recording file",
            "windowing_order": "split before windowing",
            "timestamp_policy": (
                "use post-interpolation row order/sample index; timestamps are "
                "audited but not used as window boundaries"
            ),
            "target_split_ratios": TARGET_RATIOS,
            "seed": seed,
        },
        "quality_policy": {
            "missing_accelerometer": "exclude from primary protocol",
            "missing_gyroscope": (
                "retain for accelerometer primary protocol; exclude from "
                "supplementary six-channel protocol"
            ),
            "same_subject_exact_duplicate": (
                "retain lexicographically first recording and exclude copies"
            ),
            "cross_subject_exact_duplicate": (
                "exclude every ambiguous copy to prevent user-level leakage"
            ),
            "sensor_length_mismatch": (
                "not fatal for accelerometer primary protocol; truncate to "
                "common valid length in supplementary six-channel protocol"
            ),
        },
        "raw_data_summary": {
            "official_participants": 90,
            "observed_distinct_subject_ids": len(
                {record["subject_id"] for record in records}
            ),
            "activities": len({record["activity_id"] for record in records}),
            "recording_files": len(records),
            "retained_primary_recordings": len(included),
            "excluded_primary_recordings": len(records) - len(included),
            "exclusions_by_reason": dict(sorted(exclusions.items())),
            "total_accel_samples": sum(
                record["accel_samples"] for record in records
            ),
            "total_gyro_samples": sum(
                record["gyro_samples"] for record in records
            ),
            "retained_accel_duration_hours": (
                sum(record["accel_samples"] for record in included)
                / SAMPLE_RATE_HZ
                / 3600
            ),
            "recordings_with_sensor_length_mismatch": len(mismatch_records),
            "total_absolute_sensor_length_mismatch_samples": sum(
                abs(record["accel_samples"] - record["gyro_samples"])
                for record in mismatch_records
            ),
            "recordings_with_missing_accelerometer": sum(
                record["accel_samples"] == 0 for record in records
            ),
            "recordings_with_missing_gyroscope": sum(
                record["gyro_samples"] == 0 for record in records
            ),
            "recordings_with_accel_nonmonotonic_time": sum(
                record["accel_nonmonotonic_steps"] > 0 for record in records
            ),
            "recordings_with_gyro_nonmonotonic_time": sum(
                record["gyro_nonmonotonic_steps"] > 0 for record in records
            ),
            "recordings_with_accel_duplicate_time": sum(
                record["accel_duplicate_time_steps"] > 0 for record in records
            ),
            "recordings_with_gyro_duplicate_time": sum(
                record["gyro_duplicate_time_steps"] > 0 for record in records
            ),
            "recordings_with_accel_internal_padding": sum(
                record["accel_internal_padding_rows"] > 0 for record in records
            ),
            "recordings_with_gyro_internal_padding": sum(
                record["gyro_internal_padding_rows"] > 0 for record in records
            ),
        },
        "cohort_definition": {
            "supported_recording_threshold_windows": (
                MIN_WINDOWS_PER_SUPPORTED_RECORDING
            ),
            "split_feasible_activity": (
                "at least two independent retained recordings, each with at "
                "least three complete 3-second windows"
            ),
            "full_sparse_evaluable_client": (
                "at least three split-feasible activities"
            ),
            "minimum_support_client": (
                "full-sparse evaluable, at least eight observed activities, "
                "and at least five minutes of retained accelerometer data"
            ),
            "raw_subject_ids": len(subject_rows),
            "primary_usable_clients": len(cohorts["by_subject"]),
            "clients_excluded_without_usable_recordings": (
                len(subject_rows) - len(cohorts["by_subject"])
            ),
            "subject_ids_excluded_without_usable_recordings": sorted(
                (
                    row["subject_id"]
                    for row in subject_rows
                    if row["retained_recordings"] == 0
                ),
                key=int,
            ),
            "full_sparse_evaluable_clients": len(
                cohorts["full_sparse_evaluable"]
            ),
            "minimum_support_clients": len(cohorts["minimum_support"]),
            "full_sparse_non_evaluable_clients": (
                len(cohorts["by_subject"])
                - len(cohorts["full_sparse_evaluable"])
            ),
            "minimum_support_subject_ids": sorted(
                cohorts["minimum_support"], key=int
            ),
            "minimum_support_label_ids": [
                row["activity_id"] for row in minimum_class_rows
            ],
            "minimum_support_excluded_label_ids": sorted(
                {row["activity_id"] for row in class_rows}
                - {row["activity_id"] for row in minimum_class_rows}
            ),
            "minimum_support_excluded_labels": [
                row["activity"]
                for row in class_rows
                if row["activity_id"]
                not in {item["activity_id"] for item in minimum_class_rows}
            ],
        },
        "allocation": {
            **reservation_metadata,
            **allocation_metadata,
            "split_recordings": {
                split: split_recordings[split] for split in SPLITS
            },
            "split_windows": {split: split_windows[split] for split in SPLITS},
            "split_ratios": {
                split: split_windows[split] / total_windows for split in SPLITS
            },
            "total_retained_windows": total_windows,
            "maximum_class_absolute_ratio_deviation": max(
                row["max_absolute_ratio_deviation"] for row in class_rows
            ),
            "minimum_support_split_windows": {
                split: minimum_split_windows[split] for split in SPLITS
            },
            "minimum_support_split_ratios": {
                split: minimum_split_windows[split] / minimum_total_windows
                for split in SPLITS
            },
            "minimum_support_total_windows": minimum_total_windows,
            "minimum_support_maximum_class_absolute_ratio_deviation": max(
                row["max_absolute_ratio_deviation"]
                for row in minimum_class_rows
            ),
        },
        "evaluation_support": {
            "realized_evaluable_clients": len(realized_evaluable),
            "realized_minimum_support_clients": len(minimum_realized),
            "supported_client_class_pairs": supported_pairs,
            "seen_supported_client_class_pairs": seen_pairs,
            "locally_unseen_supported_client_class_pairs": unseen_pairs,
            "minimum_support_supported_client_class_pairs": sum(
                row["supported_test_classes"] for row in minimum_rows
            ),
            "minimum_support_seen_supported_client_class_pairs": sum(
                row["seen_supported_test_classes"] for row in minimum_rows
            ),
            "minimum_support_locally_unseen_supported_client_class_pairs": sum(
                row["locally_unseen_supported_test_classes"]
                for row in minimum_rows
            ),
            "supported_test_class_threshold_windows": (
                MIN_WINDOWS_PER_SUPPORTED_RECORDING
            ),
        },
        "class_summary": class_rows,
        "minimum_support_class_summary": minimum_class_rows,
        "integrity_checks": integrity_checks,
        "artifacts": artifact_metadata,
        "implementation": {
            "deterministic_tiebreak": "SHA-256 of seed and stable identifiers",
            "deterministic_gzip_mtime": 0,
            "window_manifest_rows": window_manifest_rows,
        },
    }


def render_report(audit: dict) -> str:
    source = audit["source"]
    raw = audit["raw_data_summary"]
    cohort = audit["cohort_definition"]
    allocation = audit["allocation"]
    support = audit["evaluation_support"]
    lines = [
        "# KU-HAR Recording Split Audit V1",
        "",
        "## Decision",
        "",
        (
            "KU-HAR V5 is retained as a secondary stress dataset for "
            "protocol-induced severe label and quantity skew. The primary "
            "protocol uses the three-axis accelerometer and allocates complete "
            "recording files before constructing non-overlapping 3-second windows."
        ),
        "",
        "## Source And Integrity",
        "",
        f"- Source: Mendeley Data Version {source['version']}, DOI `{source['doi']}`.",
        f"- License: {source['license']}.",
        f"- Archive SHA-256: `{source['archive_sha256']}`.",
        (
            f"- ZIP CRC check passed; {raw['recording_files']:,} CSV recordings "
            "were scanned."
        ),
        (
            "- The official description reports 90 participants, but the V5 "
            f"archive contains {raw['observed_distinct_subject_ids']} distinct "
            "subject IDs."
        ),
        (
            f"- After quality exclusions, "
            f"{cohort['primary_usable_clients']} subject IDs retain usable "
            "accelerometer windows."
        ),
        "",
        "## Quality Findings And Frozen Policy",
        "",
        (
            f"- Retained {raw['retained_primary_recordings']:,} recordings and "
            f"excluded {raw['excluded_primary_recordings']} from the primary "
            "accelerometer protocol."
        ),
        (
            "- One recording has no accelerometer samples and is excluded; one "
            "recording has no gyroscope samples but remains eligible for the "
            "accelerometer-only primary protocol."
        ),
        (
            f"- {raw['recordings_with_sensor_length_mismatch']:,} recordings "
            "have accelerometer/gyroscope valid-length mismatch. Supplementary "
            "six-channel experiments must truncate each recording to the common "
            "valid length."
        ),
        (
            "- Same-subject exact duplicates retain only the lexicographically "
            "first member. All cross-subject exact duplicate copies are excluded "
            "to prevent user-level leakage."
        ),
        (
            "- Window boundaries use row order/sample index after interpolation, "
            "not timestamps. This avoids timestamp-quantization artifacts while "
            "preserving the stated 100 Hz sampling grid."
        ),
        "",
        "## Cohorts",
        "",
        (
            "- A split-feasible activity requires at least two independent "
            "retained recordings, each containing at least three complete "
            "3-second windows."
        ),
        (
            "- Full sparse evaluable set: "
            f"{cohort['full_sparse_evaluable_clients']}/"
            f"{cohort['primary_usable_clients']} "
            "clients with at least three split-feasible activities."
        ),
        (
            "- Minimum-support cohort: "
            f"{cohort['minimum_support_clients']} clients that are full-sparse "
            "evaluable, cover at least eight activities, and contain at least "
            "five minutes of retained accelerometer data."
        ),
        (
            "- The minimum-support cohort has a frozen 17-class label space "
            "(`0`-`16`). `Table-tennis` (`17`) occurs only among sparse clients "
            "and is evaluated in the full sparse 18-class sensitivity analysis."
        ),
        (
            f"- The remaining {cohort['full_sparse_non_evaluable_clients']} "
            "usable clients stay in the full sparse training population but are "
            "not in the primary per-user fairness denominator."
        ),
        (
            f"- {cohort['clients_excluded_without_usable_recordings']} raw "
            "subject has no usable recording after the frozen quality exclusions "
            "and is not counted as a federated client."
        ),
        "",
        "## Split Protocol And Result",
        "",
        (
            "- For every evaluable client, three activities with the strongest "
            "repeat support reserve one complete train recording and one complete "
            "test recording. No recording is cut, copied, or shared across splits."
        ),
        (
            "- Remaining recordings are assigned with deterministic joint "
            "full-cohort class, minimum-cohort class, client-total, and "
            "client-class duration balancing, using seed "
            f"`{audit['primary_protocol']['seed']}`."
        ),
        (
            "- Retained windows: "
            f"{allocation['total_retained_windows']:,}; train/validation/test = "
            f"{allocation['split_windows']['train']:,}/"
            f"{allocation['split_windows']['validation']:,}/"
            f"{allocation['split_windows']['test']:,} "
            f"({allocation['split_ratios']['train']:.2%}/"
            f"{allocation['split_ratios']['validation']:.2%}/"
            f"{allocation['split_ratios']['test']:.2%})."
        ),
        (
            "- Minimum-support windows: "
            f"{allocation['minimum_support_total_windows']:,}; "
            "train/validation/test = "
            f"{allocation['minimum_support_split_windows']['train']:,}/"
            f"{allocation['minimum_support_split_windows']['validation']:,}/"
            f"{allocation['minimum_support_split_windows']['test']:,} "
            f"({allocation['minimum_support_split_ratios']['train']:.2%}/"
            f"{allocation['minimum_support_split_ratios']['validation']:.2%}/"
            f"{allocation['minimum_support_split_ratios']['test']:.2%})."
        ),
        (
            "- Maximum class-level absolute ratio deviation: full cohort "
            f"{allocation['maximum_class_absolute_ratio_deviation']:.2%}; "
            "minimum-support cohort "
            f"{allocation['minimum_support_maximum_class_absolute_ratio_deviation']:.2%}."
        ),
        (
            f"- Realized evaluable clients: "
            f"{support['realized_evaluable_clients']}; minimum-support clients: "
            f"{support['realized_minimum_support_clients']}."
        ),
        (
            f"- Full-sparse evaluable client-class pairs: "
            f"{support['supported_client_class_pairs']}; seen: "
            f"{support['seen_supported_client_class_pairs']}; locally unseen: "
            f"{support['locally_unseen_supported_client_class_pairs']}."
        ),
        (
            f"- Minimum-support client-class pairs: "
            f"{support['minimum_support_supported_client_class_pairs']}; "
            "seen: "
            f"{support['minimum_support_seen_supported_client_class_pairs']}; "
            "locally unseen: "
            f"{support['minimum_support_locally_unseen_supported_client_class_pairs']}."
        ),
        "",
        "## Class-Level Split",
        "",
        "| ID | Activity | Windows | Train | Validation | Test | Max deviation |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit["class_summary"]:
        lines.append(
            f"| {row['activity_id']} | {row['activity']} | "
            f"{row['total_windows']:,} | {row['train_ratio']:.3f} | "
            f"{row['validation_ratio']:.3f} | {row['test_ratio']:.3f} | "
            f"{row['max_absolute_ratio_deviation']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Integrity Checks",
            "",
        ]
    )
    for name, passed in audit["integrity_checks"].items():
        lines.append(f"- `{name}`: {passed}")
    lines.extend(["", "## Frozen Artifacts", ""])
    for name, metadata in audit["artifacts"].items():
        lines.append(
            f"- {name}: `{metadata['filename']}`, SHA-256 "
            f"`{metadata['sha256']}`."
        )
    lines.extend(
        [
            "",
            (
                "Any later change to quality exclusions, cohort thresholds, "
                "allocation rules, seed, or windowing must create a new manifest "
                "version rather than overwrite V1."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        args.report
        if args.report is not None
        else args.output_dir / "kuhar_recording_split_audit_v1.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    records, scan_summary = scan_archive(args.archive)
    apply_quality_policy(records, scan_summary["duplicate_groups"])
    cohorts = derive_cohorts(records)
    reservation_metadata = reserve_support(records, cohorts, args.seed)
    allocation_metadata = allocate_recordings(records, args.seed)
    subject_rows, class_rows = summarize(records, cohorts)
    minimum_class_rows = summarize_class_subset(
        records, lambda record: record["minimum_support_client"]
    )

    recording_metadata = args.output_dir / "kuhar_recording_metadata.csv"
    recording_manifest = (
        args.output_dir / "kuhar_recording_split_manifest.csv"
    )
    window_manifest = args.output_dir / "kuhar_window_split_manifest.csv.gz"
    subject_summary = args.output_dir / "kuhar_subject_summary.csv"
    class_summary = args.output_dir / "kuhar_class_summary.csv"
    minimum_class_summary = (
        args.output_dir / "kuhar_minimum_support_class_summary.csv"
    )
    audit_json = args.output_dir / "kuhar_split_audit_v1.json"

    write_recording_metadata(recording_metadata, records)
    write_recording_manifest(recording_manifest, records)
    window_manifest_rows = write_window_manifest(window_manifest, records)
    write_csv(subject_summary, subject_rows, list(subject_rows[0]))
    write_csv(class_summary, class_rows, list(class_rows[0]))
    write_csv(
        minimum_class_summary,
        minimum_class_rows,
        list(minimum_class_rows[0]),
    )

    artifacts = {
        "recording_metadata": recording_metadata,
        "recording_split_manifest": recording_manifest,
        "window_split_manifest": window_manifest,
        "subject_summary": subject_summary,
        "class_summary": class_summary,
        "minimum_support_class_summary": minimum_class_summary,
    }
    audit = build_audit(
        records,
        scan_summary,
        cohorts,
        reservation_metadata,
        allocation_metadata,
        subject_rows,
        class_rows,
        minimum_class_rows,
        window_manifest_rows,
        args.seed,
        artifacts,
    )
    audit_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path.write_text(render_report(audit), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "report": str(report_path),
                "retained_recordings": audit["raw_data_summary"][
                    "retained_primary_recordings"
                ],
                "retained_windows": audit["allocation"][
                    "total_retained_windows"
                ],
                "full_sparse_evaluable_clients": audit["cohort_definition"][
                    "full_sparse_evaluable_clients"
                ],
                "minimum_support_clients": audit["cohort_definition"][
                    "minimum_support_clients"
                ],
                "split_ratios": audit["allocation"]["split_ratios"],
                "max_class_ratio_deviation": audit["allocation"][
                    "maximum_class_absolute_ratio_deviation"
                ],
                "integrity_checks": audit["integrity_checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
