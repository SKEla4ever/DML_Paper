# KU-HAR Recording Split Audit V1

## Decision

KU-HAR V5 is retained as a secondary stress dataset for protocol-induced severe label and quantity skew. The primary protocol uses the three-axis accelerometer and allocates complete recording files before constructing non-overlapping 3-second windows.

## Source And Integrity

- Source: Mendeley Data Version 5, DOI `10.17632/45f952y38r.5`.
- License: CC BY 4.0.
- Archive SHA-256: `9fe5d0052f2f1d6711afac42ee4badd968116afa8ba4b8ba591f4fdd771c2ec2`.
- ZIP CRC check passed; 1,945 CSV recordings were scanned.
- The official description reports 90 participants, but the V5 archive contains 89 distinct subject IDs.
- After quality exclusions, 88 subject IDs retain usable accelerometer windows.

## Quality Findings And Frozen Policy

- Retained 1,938 recordings and excluded 7 from the primary accelerometer protocol.
- One recording has no accelerometer samples and is excluded; one recording has no gyroscope samples but remains eligible for the accelerometer-only primary protocol.
- 752 recordings have accelerometer/gyroscope valid-length mismatch. Supplementary six-channel experiments must truncate each recording to the common valid length.
- Same-subject exact duplicates retain only the lexicographically first member. All cross-subject exact duplicate copies are excluded to prevent user-level leakage.
- Window boundaries use row order/sample index after interpolation, not timestamps. This avoids timestamp-quantization artifacts while preserving the stated 100 Hz sampling grid.

## Cohorts

- A split-feasible activity requires at least two independent retained recordings, each containing at least three complete 3-second windows.
- Full sparse evaluable set: 54/88 clients with at least three split-feasible activities.
- Minimum-support cohort: 50 clients that are full-sparse evaluable, cover at least eight activities, and contain at least five minutes of retained accelerometer data.
- The minimum-support cohort has a frozen 17-class label space (`0`-`16`). `Table-tennis` (`17`) occurs only among sparse clients and is evaluated in the full sparse 18-class sensitivity analysis.
- The remaining 34 usable clients stay in the full sparse training population but are not in the primary per-user fairness denominator.
- 1 raw subject has no usable recording after the frozen quality exclusions and is not counted as a federated client.

## Split Protocol And Result

- For every evaluable client, three activities with the strongest repeat support reserve one complete train recording and one complete test recording. No recording is cut, copied, or shared across splits.
- Remaining recordings are assigned with deterministic joint full-cohort class, minimum-cohort class, client-total, and client-class duration balancing, using seed `20260615`.
- Retained windows: 19,641; train/validation/test = 12,351/3,578/3,712 (62.88%/18.22%/18.90%).
- Minimum-support windows: 13,627; train/validation/test = 8,177/2,632/2,818 (60.01%/19.31%/20.68%).
- Maximum class-level absolute ratio deviation: full cohort 5.79%; minimum-support cohort 6.49%.
- Realized evaluable clients: 54; minimum-support clients: 50.
- Full-sparse evaluable client-class pairs: 279; seen: 244; locally unseen: 35.
- Minimum-support client-class pairs: 264; seen: 229; locally unseen: 35.

## Class-Level Split

| ID | Activity | Windows | Train | Validation | Test | Max deviation |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Stand | 1,830 | 0.658 | 0.166 | 0.176 | 0.058 |
| 1 | Sit | 1,837 | 0.652 | 0.171 | 0.177 | 0.052 |
| 2 | Talk-sit | 1,733 | 0.657 | 0.168 | 0.175 | 0.057 |
| 3 | Talk-stand | 1,819 | 0.651 | 0.172 | 0.177 | 0.051 |
| 4 | Stand-sit | 2,019 | 0.598 | 0.202 | 0.200 | 0.002 |
| 5 | Lay | 1,771 | 0.656 | 0.170 | 0.174 | 0.056 |
| 6 | Lay-stand | 1,685 | 0.591 | 0.195 | 0.214 | 0.014 |
| 7 | Pick | 1,281 | 0.646 | 0.167 | 0.187 | 0.046 |
| 8 | Jump | 610 | 0.605 | 0.195 | 0.200 | 0.005 |
| 9 | Push-up | 425 | 0.598 | 0.202 | 0.200 | 0.002 |
| 10 | Sit-up | 944 | 0.606 | 0.198 | 0.196 | 0.006 |
| 11 | Walk | 786 | 0.601 | 0.200 | 0.200 | 0.001 |
| 12 | Walk-backwards | 289 | 0.612 | 0.194 | 0.194 | 0.012 |
| 13 | Walk-circle | 238 | 0.622 | 0.189 | 0.189 | 0.022 |
| 14 | Run | 485 | 0.614 | 0.177 | 0.208 | 0.023 |
| 15 | Stair-up | 777 | 0.605 | 0.202 | 0.193 | 0.007 |
| 16 | Stair-down | 754 | 0.609 | 0.198 | 0.194 | 0.009 |
| 17 | Table-tennis | 358 | 0.601 | 0.176 | 0.223 | 0.024 |

## Integrity Checks

- `all_retained_recordings_assigned_once`: True
- `all_excluded_recordings_marked_excluded`: True
- `window_manifest_matches_retained_window_count`: True
- `every_usable_client_has_train_data`: True
- `all_frozen_evaluable_clients_realized`: True
- `all_minimum_support_clients_realized`: True
- `cross_subject_exact_duplicates_excluded`: True
- `all_classes_present_in_all_splits`: True
- `all_minimum_cohort_classes_present_in_all_splits`: True

## Frozen Artifacts

- recording_metadata: `kuhar_recording_metadata.csv`, SHA-256 `4ae276289fe0d7f8f3720f13c60512941dc87d3f427ea0262180f64c7a0e1e6f`.
- recording_split_manifest: `kuhar_recording_split_manifest.csv`, SHA-256 `d06d169fa514695c444e15b9e9663d71472e82704e12b6250bf4715a7cdd65e3`.
- window_split_manifest: `kuhar_window_split_manifest.csv.gz`, SHA-256 `ee2c939254bee8635bcd6443417208d9d4ec9e3c9dcef932bcb28faec460499e`.
- subject_summary: `kuhar_subject_summary.csv`, SHA-256 `95834d80c19431f294e7565e051edbee59c83fabec5540ef16fcaa5eb59e7e29`.
- class_summary: `kuhar_class_summary.csv`, SHA-256 `e52fb8cd46c50e70ebd31c9c8a330c64c2c29616763ad63f2ef6bf948a016e9a`.
- minimum_support_class_summary: `kuhar_minimum_support_class_summary.csv`, SHA-256 `22e0ee0e42f98ccca4ccc42262c37e7e4bf15d5a6f87cf4df7b5fdb2b0b4ecac`.

Any later change to quality exclusions, cohort thresholds, allocation rules, seed, or windowing must create a new manifest version rather than overwrite V1.
