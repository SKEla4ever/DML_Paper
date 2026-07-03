# KU-HAR 1D CNN FedAvg Tuning V1

## Purpose

This report records the first validation-driven tuning pass for a standard
neural FedAvg baseline on the frozen KU-HAR V1 minimum-support cohort.

Selection rule: choose configuration by validation Macro-F1, with validation
per-user mean Macro-F1 as secondary context. Test metrics are recorded for
audit visibility but are not used to choose configurations.

## Fixed Protocol

- Cohort: minimum-support
- Clients: 50
- Labels: 17 (`0`-`16`)
- Client fraction: 1.0
- Local epochs: 1 for the pilot grid, then 2 for targeted sensitivity
- Batch size: 64
- Seed: `20260615`
- Input: raw 3-axis accelerometer windows, `3 x 300`
- Model family: 1D CNN

## Pilot Grid

All pilot grid runs use 20 rounds and `eval_every=5`.

| Rank | Normalization | Optimizer | Learning rate | Momentum | Validation Macro-F1 | Validation user mean Macro-F1 | Total communication |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | BatchNorm | Adam | 0.001 | 0.0 | 0.2770 | 0.4461 | 199,608,000 |
| 2 | BatchNorm | SGD | 0.030 | 0.9 | 0.2201 | 0.3969 | 199,608,000 |
| 3 | BatchNorm | Adam | 0.0003 | 0.0 | 0.1888 | 0.3049 | 199,608,000 |
| 4 | GroupNorm | SGD | 0.030 | 0.9 | 0.1833 | 0.3020 | 197,000,000 |
| 5 | GroupNorm | Adam | 0.001 | 0.0 | 0.1785 | 0.2999 | 197,000,000 |
| 6 | BatchNorm | SGD | 0.010 | 0.9 | 0.1460 | 0.2509 | 199,608,000 |
| 7 | GroupNorm | SGD | 0.010 | 0.9 | 0.1278 | 0.2158 | 197,000,000 |
| 8 | GroupNorm | Adam | 0.0003 | 0.0 | 0.1090 | 0.2287 | 197,000,000 |

Pilot conclusion: BatchNorm with Adam at learning rate `0.001` is the best
20-round configuration by validation Macro-F1. GroupNorm improves neither
Macro-F1 nor user-level Macro-F1 in this first grid.

## Extended Candidate

The best pilot configuration was extended to 50 and 100 rounds:

| Rounds | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Total communication |
|---:|---:|---:|---:|---:|
| 20 | 0.2770 | 0.4461 | 0.1815 | 199,608,000 |
| 50 | 0.3281 | 0.4963 | 0.2630 | 499,020,000 |
| 100 | 0.4178 | 0.5748 | 0.3330 | 998,040,000 |

100-round trajectory:

| Round | Validation Macro-F1 | Validation user mean Macro-F1 | Total communication |
|---:|---:|---:|---:|
| 0 | 0.0282 | 0.0530 | 0 |
| 10 | 0.1908 | 0.3066 | 99,804,000 |
| 20 | 0.2770 | 0.4461 | 199,608,000 |
| 30 | 0.2934 | 0.4671 | 299,412,000 |
| 40 | 0.3046 | 0.4777 | 399,216,000 |
| 50 | 0.3281 | 0.4963 | 499,020,000 |
| 60 | 0.3597 | 0.5075 | 598,824,000 |
| 70 | 0.3677 | 0.5106 | 698,628,000 |
| 80 | 0.3924 | 0.5382 | 798,432,000 |
| 90 | 0.3847 | 0.5407 | 898,236,000 |
| 100 | 0.4178 | 0.5748 | 998,040,000 |

## Local Epoch Sensitivity

The best normalization/optimizer/learning-rate setting was also tested with
local epochs `2`.

| Local epochs | Rounds | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Total communication |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.2770 | 0.4461 | 0.1815 | 199,608,000 |
| 2 | 20 | 0.3042 | 0.4763 | 0.2306 | 199,608,000 |
| 1 | 50 | 0.3281 | 0.4963 | 0.2630 | 499,020,000 |
| 2 | 50 | 0.3964 | 0.5567 | 0.3211 | 499,020,000 |
| 1 | 100 | 0.4178 | 0.5748 | 0.3330 | 998,040,000 |

Local epochs `2` substantially improves communication efficiency in this first
pass. At 50 rounds it reaches validation Macro-F1 `0.3964` with about half the
communication of the 100-round local-epoch-1 run.

## Current Baseline Candidate

Use the following as the current high-budget FedAvg baseline candidate:

- Normalization: BatchNorm
- Optimizer: Adam
- Learning rate: 0.001
- Local epochs: 1
- Batch size: 64
- Rounds: 100

Current final test snapshot for this candidate:

- Test accuracy: 0.5806
- Test Macro-F1: 0.4300
- Test per-user mean Macro-F1: 0.6369
- Test worst 10% per-user Macro-F1: 0.3406
- Total communication: 998,040,000 bytes

Use the following as the current communication-efficient FedAvg baseline
candidate:

- Normalization: BatchNorm
- Optimizer: Adam
- Learning rate: 0.001
- Local epochs: 2
- Batch size: 64
- Rounds: 50

Current final test snapshot for this candidate:

- Test accuracy: 0.5880
- Test Macro-F1: 0.4181
- Test per-user mean Macro-F1: 0.6289
- Test worst 10% per-user Macro-F1: 0.3175
- Total communication: 499,020,000 bytes

## Interpretation

The first neural checkpoint at 20 rounds was under-trained. Extending the best
configuration to 100 rounds substantially improves validation Macro-F1 and
per-user fairness metrics. The 100-round validation curve is still improving,
so the next decision is whether the project fixes `100 rounds` as the common
communication budget checkpoint or adds a longer-budget sensitivity run.

Recommended next step: use `BatchNorm + Adam lr=0.001 + local epochs 2` as the
working communication-efficient FedAvg baseline setting, and compare methods
at shared communication checkpoints. Run a longer local-epoch-2 endpoint only
if the paper needs a high-budget FedAvg upper bound.
