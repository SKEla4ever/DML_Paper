#!/usr/bin/env python3
"""Shared raw-data primitives for the frozen HHAR phone-accelerometer pipeline."""

from __future__ import annotations

import hashlib
import math
import statistics
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


DATASET_DOI = "10.24432/C5689X"
DATASET_URL = (
    "https://archive.ics.uci.edu/dataset/344/"
    "heterogeneity%2Bactivity%2Brecognition"
)
DATASET_LICENSE = "CC BY 4.0"
EXPECTED_ACTIVITY_ARCHIVE_SHA256 = (
    "d4c0c53b195b523859bf71f5a349d164c7a604a321ff6b0972fbed6e03b46582"
)
PHONE_ACCELEROMETER_MEMBER = "Activity recognition exp/Phones_accelerometer.csv"
EXPECTED_HEADER = (
    b"Index,Arrival_Time,Creation_Time,x,y,z,User,Model,Device,gt"
)
ACTIVITIES = ("bike", "sit", "stand", "walk", "stairsup", "stairsdown")
ACTIVITY_TO_ID = {activity: index for index, activity in enumerate(ACTIVITIES)}
TARGET_SAMPLE_RATE_HZ = 50
TARGET_INTERVAL_NS = 1_000_000_000 // TARGET_SAMPLE_RATE_HZ
WINDOW_SECONDS = 3
WINDOW_SAMPLES = TARGET_SAMPLE_RATE_HZ * WINDOW_SECONDS
WINDOW_DURATION_NS = WINDOW_SECONDS * 1_000_000_000
MAX_INTERPOLATION_GAP_NS = 1_000_000_000


@dataclass
class LabelRun:
    user_id: str
    model: str
    device_id: str
    activity: str
    activity_ordinal: int
    first_source_line: int
    last_source_line: int
    arrival_times_ms: list[int]
    creation_times_ns: list[int]
    values: list[tuple[float, float, float]] | None
    content_sha256: str

    @property
    def run_id(self) -> str:
        return (
            f"user={self.user_id}/device={self.device_id}/"
            f"activity={self.activity}/bout={self.activity_ordinal:02d}"
        )

    @property
    def client_id(self) -> str:
        return f"user={self.user_id}|device={self.device_id}"


@dataclass
class ContinuousSegment:
    run: LabelRun
    segment_index: int
    creation_times_ns: list[int]
    arrival_times_ms: list[int]
    values: list[tuple[float, float, float]] | None
    gap_before_ns: int | None
    duplicate_creation_timestamps: int

    @property
    def recording_id(self) -> str:
        return f"{self.run.run_id}/segment={self.segment_index:03d}"

    @property
    def start_creation_ns(self) -> int:
        return self.creation_times_ns[0]

    @property
    def end_creation_ns(self) -> int:
        return self.creation_times_ns[-1]

    @property
    def duration_ns(self) -> int:
        return self.end_creation_ns - self.start_creation_ns

    @property
    def window_count(self) -> int:
        return int((self.duration_ns + TARGET_INTERVAL_NS) // WINDOW_DURATION_NS)

    @property
    def median_interval_ns(self) -> float | None:
        if len(self.creation_times_ns) < 2:
            return None
        differences = [
            current - previous
            for previous, current in zip(
                self.creation_times_ns, self.creation_times_ns[1:]
            )
        ]
        return float(statistics.median(differences))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_activity_archive(path: Path, *, test_crc: bool) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"HHAR activity archive not found: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != EXPECTED_ACTIVITY_ARCHIVE_SHA256:
        raise RuntimeError(
            "archive SHA-256 does not match the frozen HHAR source: "
            f"{actual_sha256}"
        )
    with zipfile.ZipFile(path) as archive:
        if PHONE_ACCELEROMETER_MEMBER not in archive.namelist():
            raise RuntimeError(
                f"missing {PHONE_ACCELEROMETER_MEMBER!r} in {path}"
            )
        bad_member = archive.testzip() if test_crc else None
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC validation failed at {bad_member}")
        member = archive.getinfo(PHONE_ACCELEROMETER_MEMBER)
    return {
        "archive_sha256": actual_sha256,
        "archive_size_bytes": path.stat().st_size,
        "zip_crc_check_passed": bool(test_crc),
        "phone_accelerometer_member": PHONE_ACCELEROMETER_MEMBER,
        "phone_accelerometer_uncompressed_bytes": member.file_size,
        "phone_accelerometer_crc32": f"{member.CRC:08x}",
    }


def iter_label_runs(
    archive_path: Path, *, include_values: bool
) -> Iterator[LabelRun]:
    """Yield source-order label runs; the frozen CSV is client-contiguous."""

    activity_ordinals: Counter[tuple[str, str, str]] = Counter()
    closed_clients: set[tuple[str, str]] = set()
    current_client: tuple[str, str] | None = None
    current_label: str | None = None
    current_model = ""
    arrivals: list[int] = []
    creations: list[int] = []
    values: list[tuple[float, float, float]] | None = [] if include_values else None
    first_line = 0
    last_line = 0
    digest = hashlib.sha256()

    def flush() -> LabelRun | None:
        nonlocal arrivals, creations, values, digest
        if current_client is None or current_label is None or not arrivals:
            return None
        user_id, device_id = current_client
        ordinal_key = (user_id, device_id, current_label)
        activity_ordinals[ordinal_key] += 1
        run = LabelRun(
            user_id=user_id,
            model=current_model,
            device_id=device_id,
            activity=current_label,
            activity_ordinal=activity_ordinals[ordinal_key],
            first_source_line=first_line,
            last_source_line=last_line,
            arrival_times_ms=arrivals,
            creation_times_ns=creations,
            values=values,
            content_sha256=digest.hexdigest(),
        )
        arrivals = []
        creations = []
        values = [] if include_values else None
        digest = hashlib.sha256()
        return run

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(PHONE_ACCELEROMETER_MEMBER) as handle:
            header = handle.readline().rstrip(b"\r\n")
            if header != EXPECTED_HEADER:
                raise RuntimeError(f"unexpected HHAR CSV header: {header!r}")
            for source_line, raw_line in enumerate(handle, start=2):
                fields = raw_line.rstrip(b"\r\n").split(b",")
                if len(fields) != 10:
                    raise RuntimeError(
                        f"malformed HHAR row at source line {source_line}: "
                        f"expected 10 fields, found {len(fields)}"
                    )
                try:
                    arrival = int(fields[1])
                    creation = int(fields[2])
                    axes = tuple(float(fields[index]) for index in (3, 4, 5))
                except ValueError as exc:
                    raise RuntimeError(
                        f"invalid numeric value at source line {source_line}"
                    ) from exc
                if not all(math.isfinite(axis) for axis in axes):
                    raise RuntimeError(
                        f"non-finite accelerometer value at source line {source_line}"
                    )
                user_id = fields[6].decode("utf-8")
                model = fields[7].decode("utf-8")
                device_id = fields[8].decode("utf-8")
                activity = fields[9].decode("utf-8").strip().lower()
                client = (user_id, device_id)

                if current_client is not None and client != current_client:
                    completed = flush()
                    if completed is not None:
                        yield completed
                    closed_clients.add(current_client)
                    if client in closed_clients:
                        raise RuntimeError(
                            "HHAR CSV is not client-contiguous; preprocessing "
                            f"contract violated by {client}"
                        )
                    current_label = None

                if current_label is not None and activity != current_label:
                    completed = flush()
                    if completed is not None:
                        yield completed

                if not arrivals:
                    first_line = source_line
                current_client = client
                current_label = activity
                current_model = model
                last_line = source_line
                arrivals.append(arrival)
                creations.append(creation)
                if values is not None:
                    values.append(axes)
                digest.update(raw_line)

    completed = flush()
    if completed is not None:
        yield completed


def split_continuous_segments(run: LabelRun) -> list[ContinuousSegment]:
    """Sort sensor timestamps and cut gaps that must not be interpolated."""

    order = sorted(range(len(run.creation_times_ns)), key=run.creation_times_ns.__getitem__)
    times: list[int] = []
    arrivals: list[int] = []
    values: list[tuple[float, float, float]] | None = [] if run.values is not None else None
    duplicate_count = 0
    for source_index in order:
        creation = run.creation_times_ns[source_index]
        if times and creation == times[-1]:
            duplicate_count += 1
            continue
        times.append(creation)
        arrivals.append(run.arrival_times_ms[source_index])
        if values is not None and run.values is not None:
            values.append(run.values[source_index])

    boundaries = [0]
    gaps_before: dict[int, int] = {}
    for index in range(1, len(times)):
        gap = times[index] - times[index - 1]
        if gap > MAX_INTERPOLATION_GAP_NS:
            boundaries.append(index)
            gaps_before[index] = gap
    boundaries.append(len(times))

    segments = []
    for segment_index, (start, end) in enumerate(
        zip(boundaries, boundaries[1:]), start=1
    ):
        if start == end:
            continue
        segments.append(
            ContinuousSegment(
                run=run,
                segment_index=segment_index,
                creation_times_ns=times[start:end],
                arrival_times_ms=arrivals[start:end],
                values=values[start:end] if values is not None else None,
                gap_before_ns=gaps_before.get(start),
                duplicate_creation_timestamps=duplicate_count,
            )
        )
    return segments


def resample_windows(
    segment: ContinuousSegment, window_starts_ns: list[int]
) -> np.ndarray:
    if segment.values is None:
        raise ValueError("raw values are required for resampling")
    if not window_starts_ns:
        return np.empty((0, 3, WINDOW_SAMPLES), dtype=np.float32)

    origin = segment.start_creation_ns
    source_times = np.asarray(segment.creation_times_ns, dtype=np.int64) - origin
    source_values = np.asarray(segment.values, dtype=np.float64)
    result = np.empty((len(window_starts_ns), 3, WINDOW_SAMPLES), dtype=np.float32)
    sample_offsets = np.arange(WINDOW_SAMPLES, dtype=np.int64) * TARGET_INTERVAL_NS
    for window_index, absolute_start in enumerate(window_starts_ns):
        targets = absolute_start - origin + sample_offsets
        if targets[0] < source_times[0] or targets[-1] > source_times[-1]:
            raise RuntimeError(
                f"window target lies outside {segment.recording_id}: "
                f"{absolute_start}"
            )
        for axis in range(3):
            result[window_index, axis] = np.interp(
                targets, source_times, source_values[:, axis]
            ).astype(np.float32)
    return result
