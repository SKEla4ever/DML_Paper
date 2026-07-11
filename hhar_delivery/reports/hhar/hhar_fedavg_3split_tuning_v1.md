# HHAR FedAvg Three-Split Validation-Only Tuning V1

## Selection Boundary

This staged pass selects a standard global FedAvg baseline using only validation metrics aggregated over the three frozen execution splits. No test prediction, test performance metric, or test-driven checkpoint choice is produced during tuning.

## Pre-Registered Protocol

- Split seeds: `[20260615, 20260616, 20260617]`.
- Fixed model/optimizer seed: `20260615`.
- Pilot: 8 frozen normalization/optimizer/learning-rate configs, 20 rounds, 1 local epoch, full participation.
- Practical-tie tolerance: `0.005` validation Macro-F1.
- Local computation: compare 1 versus 2 local epochs only for the pilot winner.
- Final endpoint: fixed 50 rounds; trajectory is descriptive and does not select a checkpoint.
- Global validation Macro-F1 is the selection metric. Pair-level values are descriptive because 69 user-device pairs come from nine users.

## Pilot Grid

| Selected | Config | Norm | Optimizer | LR | Val mean +/- SD | Worst split | Range | Communication |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yes | batchnorm_sgd_lr0p03 | batchnorm | sgd | 0.03 | 0.5396 +/- 0.0163 | 0.5276 | 0.0306 | 267,565,440 |
|  | batchnorm_sgd_lr0p01 | batchnorm | sgd | 0.01 | 0.4649 +/- 0.0073 | 0.4574 | 0.0145 | 267,565,440 |
|  | batchnorm_adam_lr0p0003 | batchnorm | adam | 0.0003 | 0.4430 +/- 0.0134 | 0.4277 | 0.0251 | 267,565,440 |
|  | groupnorm_adam_lr0p001 | groupnorm | adam | 0.001 | 0.4277 +/- 0.0090 | 0.4174 | 0.0160 | 263,966,400 |
|  | batchnorm_adam_lr0p001 | batchnorm | adam | 0.001 | 0.4120 +/- 0.0162 | 0.3934 | 0.0295 | 267,565,440 |
|  | groupnorm_sgd_lr0p03 | groupnorm | sgd | 0.03 | 0.3222 +/- 0.0122 | 0.3138 | 0.0225 | 263,966,400 |
|  | groupnorm_adam_lr0p0003 | groupnorm | adam | 0.0003 | 0.3174 +/- 0.0053 | 0.3132 | 0.0102 | 263,966,400 |
|  | groupnorm_sgd_lr0p01 | groupnorm | sgd | 0.01 | 0.2470 +/- 0.0060 | 0.2405 | 0.0118 | 263,966,400 |

Practical-tie set: `['batchnorm_sgd_lr0p03']`. Selected pilot config: `batchnorm_sgd_lr0p03`.

## Local-Epoch Sensitivity

| Local epochs | Val mean +/- SD | Worst split | Range | Communication | Selected |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.5396 +/- 0.0163 | 0.5276 | 0.0306 | 267,565,440 | yes |
| 2 | 0.5226 +/- 0.0080 | 0.5153 | 0.0158 | 267,565,440 |  |

Mean improvement from 1 to 2 local epochs: `-0.0170`; worst-split change: `-0.0123`. Selected local epochs: `1`.

## Frozen 50-Round Endpoint

- Configuration: `batchnorm_sgd_lr0p03`.
- Local epochs: `1`.
- Aggregate validation Macro-F1: `0.3910 +/- 0.0278`.
- Validation range: `0.0555`.
- Communication: `668,913,600` bytes.

| Split seed | Validation Macro-F1 | Communication |
| --- | --- | --- |
| 20260615 | 0.3932 | 668,913,600 |
| 20260616 | 0.4176 | 668,913,600 |
| 20260617 | 0.3621 | 668,913,600 |

## Validation Trajectory

| Round | Val mean +/- SD | Worst split | Communication |
| --- | --- | --- | --- |
| 0 | 0.0308 +/- 0.0001 | 0.0307 | 0 |
| 5 | 0.4837 +/- 0.0063 | 0.4764 | 66,891,360 |
| 10 | 0.4817 +/- 0.0138 | 0.4669 | 133,782,720 |
| 15 | 0.4690 +/- 0.0212 | 0.4476 | 200,674,080 |
| 20 | 0.5396 +/- 0.0163 | 0.5276 | 267,565,440 |
| 25 | 0.4828 +/- 0.0377 | 0.4492 | 334,456,800 |
| 30 | 0.4840 +/- 0.0251 | 0.4650 | 401,348,160 |
| 35 | 0.4394 +/- 0.0214 | 0.4215 | 468,239,520 |
| 40 | 0.3706 +/- 0.0521 | 0.3370 | 535,130,880 |
| 45 | 0.3942 +/- 0.0315 | 0.3631 | 602,022,240 |
| 50 | 0.3910 +/- 0.0278 | 0.3621 | 668,913,600 |

## Validation By Device

These values are descriptive and were not used for selection.

| Device | Val mean +/- SD | Range |
| --- | --- | --- |
| nexus4_1 | 0.4774 +/- 0.0278 | 0.0484 |
| nexus4_2 | 0.4319 +/- 0.0502 | 0.0931 |
| s3_1 | 0.3191 +/- 0.0339 | 0.0637 |
| s3_2 | 0.3693 +/- 0.0377 | 0.0717 |
| s3mini_1 | 0.3197 +/- 0.0353 | 0.0622 |
| s3mini_2 | 0.2663 +/- 0.0344 | 0.0640 |
| samsungold_1 | 0.4340 +/- 0.0192 | 0.0357 |
| samsungold_2 | 0.2627 +/- 0.0017 | 0.0034 |

## Interpretation

The fixed 50-round endpoint changes mean validation Macro-F1 by `-0.1486` relative to its independently reproduced round-20 checkpoint. The endpoint remains fixed regardless of whether an earlier trajectory point is numerically higher.

Every machine-readable training history contains exactly `train` and `validation` metrics. The tuning code rejects any run containing a test metric, so the selected setting remains test-blind.

This is split sensitivity over only nine physical users, not an estimate based on 69 independent users. Model-seed sensitivity must be run separately after freezing this configuration.

## Artifacts

- Local run results: `outputs/hhar_fedavg_3split_tuning_v1/run_results.csv`
- Local pilot summary: `outputs/hhar_fedavg_3split_tuning_v1/pilot_summary.csv`
- Published summary: `hhar_delivery/reports/hhar/hhar_fedavg_3split_tuning_v1_summary.json`
