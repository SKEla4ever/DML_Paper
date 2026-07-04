# KU-HAR FedProx V1

## Purpose

This report records the first FedProx implementation and tuning pass on the
frozen KU-HAR V1 minimum-support cohort.

FedProx keeps the same 1D-CNN backbone, frozen manifest loader, metrics, and
communication accounting as the tuned FedAvg baseline. The only algorithmic
change is the local proximal objective:

`loss = cross_entropy + (mu / 2) * ||w_local - w_global||^2`

Selection uses validation Macro-F1, with validation per-user mean Macro-F1 and
worst 10% user Macro-F1 as secondary context. Test metrics are recorded for
audit visibility but are not used to choose `mu`.

## Fixed Setting

- Cohort: minimum-support
- Clients: 50
- Labels: 17 (`0`-`16`)
- Normalization: BatchNorm
- Optimizer: Adam
- Learning rate: 0.001
- Local epochs: 2
- Batch size: 64
- Client fraction: 1.0
- Seed: `20260615`

## 20-Round Mu Grid

| Rank | Mu | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Test Macro-F1 snapshot | Total communication |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1 | 0.3045 | 0.4761 | 0.2294 | 0.3027 | 199,608,000 |
| 2 | 0.0 | 0.3042 | 0.4763 | 0.2306 | 0.3023 | 199,608,000 |
| 3 | 0.001 | 0.3042 | 0.4763 | 0.2306 | 0.3027 | 199,608,000 |
| 4 | 0.01 | 0.3038 | 0.4760 | 0.2306 | 0.3031 | 199,608,000 |

The 20-round grid shows no meaningful separation between FedProx and the
`mu=0` FedAvg-equivalent control. `mu=0.1` has the highest validation Macro-F1
by a very small margin, but slightly worse user-level validation metrics.

## 50-Round Candidate

The best 20-round `mu` by validation Macro-F1 (`mu=0.1`) was extended to 50
rounds and compared to the tuned FedAvg communication-efficient candidate.

| Method | Mu | Rounds | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Test Macro-F1 snapshot | Test user mean Macro-F1 snapshot | Total communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 0.0 | 50 | 0.3964 | 0.5567 | 0.3211 | 0.4181 | 0.6289 | 499,020,000 |
| FedProx | 0.1 | 50 | 0.3970 | 0.5554 | 0.3053 | 0.4158 | 0.6269 | 499,020,000 |

## Interpretation

FedProx V1 is implemented and validated, but this first pass does not show a
meaningful improvement over the tuned FedAvg baseline on KU-HAR
minimum-support. The tiny validation Macro-F1 gain for `mu=0.1` at 50 rounds is
not accompanied by better per-user fairness metrics or test snapshot metrics.

Current conclusion for the experiment matrix:

- FedProx should be included as a heterogeneity-correction baseline.
- Use `mu=0.1` as the selected FedProx V1 setting if a single FedProx variant
  is needed, because it is the top validation Macro-F1 setting in the pilot.
- Report that FedProx is essentially tied with tuned FedAvg in this setting,
  not clearly better.

Recommended next algorithm: SCAFFOLD, because it targets client drift more
directly than FedProx and may behave differently under KU-HAR protocol-induced
label and quantity skew.
