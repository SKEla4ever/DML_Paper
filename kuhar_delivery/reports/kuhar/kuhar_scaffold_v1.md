# KU-HAR SCAFFOLD V1

## Purpose

This report records the first SCAFFOLD implementation and tuning pass on the
frozen KU-HAR V1 minimum-support cohort.

SCAFFOLD keeps the same 1D-CNN backbone, frozen manifest loader, metrics, and
split protocol as the tuned FedAvg/FedProx baselines. The local optimizer is a
corrected SGD-style update:

`w <- w - lr * (grad - c_client + c_server)`

Client control variates use the standard option-II update:

`c_client_new = c_client - c_server + (w_global - w_local) / (local_steps * lr)`

Selection uses validation Macro-F1, with validation per-user mean Macro-F1 and
worst 10% user Macro-F1 as secondary context. Test metrics are recorded as
snapshots for audit visibility but are not used for selection.

## Fixed Setting

- Cohort: minimum-support
- Clients: 50
- Labels: 17 (`0`-`16`)
- Normalization: BatchNorm
- Local optimizer: corrected SGD
- Local epochs: 2
- Batch size: 64
- Client fraction: 1.0
- Seed: `20260615`

## Communication Accounting

SCAFFOLD accounts for both model-state exchange and control-variate exchange.

- Bytes per model state: 99,804
- Bytes per control variate: 98,500
- Per round with 50 clients: 19,830,400 bytes
- 20 rounds: 396,608,000 bytes
- 50 rounds: 991,520,000 bytes

This is roughly twice the communication of FedAvg/FedProx for the same number
of rounds in this implementation.

## 20-Round Learning-Rate Grid

All grid runs use 20 rounds and `eval_every=5`.

| Rank | LR | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Test Macro-F1 snapshot | Total communication |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.0 | 0.4329 | 0.6000 | 0.3648 | 0.4159 | 396,608,000 |
| 2 | 0.5 | 0.4269 | 0.5852 | 0.3255 | 0.4243 | 396,608,000 |
| 3 | 2.0 | 0.4250 | 0.5406 | 0.2628 | 0.4053 | 396,608,000 |
| 4 | 0.8 | 0.4189 | 0.5821 | 0.3278 | 0.4195 | 396,608,000 |
| 5 | 1.5 | 0.4003 | 0.5423 | 0.2987 | 0.3814 | 396,608,000 |
| 6 | 0.3 | 0.3633 | 0.5338 | 0.2698 | 0.3606 | 396,608,000 |
| 7 | 0.2 | 0.3572 | 0.5120 | 0.3020 | 0.3516 | 396,608,000 |
| 8 | 0.1 | 0.2939 | 0.4568 | 0.2002 | 0.2779 | 396,608,000 |
| 9 | 0.05 | 0.1990 | 0.3846 | 0.1379 | 0.1791 | 396,608,000 |
| 10 | 0.03 | 0.1426 | 0.3246 | 0.1042 | 0.1359 | 396,608,000 |
| 11 | 0.01 | 0.0790 | 0.1978 | 0.0000 | 0.0635 | 396,608,000 |
| 12 | 0.003 | 0.0326 | 0.1144 | 0.0000 | 0.0350 | 396,608,000 |
| 13 | 0.001 | 0.0312 | 0.1129 | 0.0000 | 0.0303 | 396,608,000 |

The early low-learning-rate grid under-trained badly. Corrected SGD needed a
much larger learning rate than the Adam-based FedAvg/FedProx baselines.
Validation Macro-F1 peaked at `lr=1.0`; `lr=1.5` dropped clearly, and `lr=2.0`
partially recovered but had weaker user-level validation metrics than `lr=1.0`.

## 50-Round Candidate

The best 20-round learning rate by validation Macro-F1 (`lr=1.0`) was extended
to 50 rounds.

| Round | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Test Macro-F1 snapshot | Total communication |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0282 | 0.0530 | 0.0000 | 0.0292 | 0 |
| 10 | 0.3062 | 0.4781 | 0.2531 | 0.2899 | 198,304,000 |
| 20 | 0.4329 | 0.6000 | 0.3648 | 0.4159 | 396,608,000 |
| 30 | 0.4728 | 0.6232 | 0.3399 | 0.4851 | 594,912,000 |
| 40 | 0.5289 | 0.6606 | 0.3664 | 0.5379 | 793,216,000 |
| 50 | 0.5925 | 0.7041 | 0.4478 | 0.5952 | 991,520,000 |

## Baseline Comparison

| Method | Local optimizer | Key hyperparameter | Rounds | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Test Macro-F1 snapshot | Test user mean Macro-F1 snapshot | Total communication |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | Adam | lr=0.001 | 50 | 0.3964 | 0.5567 | 0.3211 | 0.4181 | 0.6289 | 499,020,000 |
| FedProx | Adam | mu=0.1 | 50 | 0.3970 | 0.5554 | 0.3053 | 0.4158 | 0.6269 | 499,020,000 |
| FedAvg | Adam | lr=0.001 | 100 | 0.4178 | 0.5748 | 0.3330 | 0.4300 | 0.6369 | 998,040,000 |
| SCAFFOLD | corrected SGD | lr=1.0 | 50 | 0.5925 | 0.7041 | 0.4478 | 0.5952 | 0.7226 | 991,520,000 |

At the same number of rounds, SCAFFOLD is not communication-matched because it
exchanges control variates as well as model states. At a similar communication
budget, SCAFFOLD 50 rounds still substantially outperforms FedAvg 100 rounds in
this first pass.

## Interpretation

SCAFFOLD V1 is implemented, smoke-tested, and tuned enough to serve as the
current client-drift baseline. Unlike FedProx V1, it shows a large improvement
on the KU-HAR minimum-support protocol.

Current conclusion for the experiment matrix:

- Use `BatchNorm + corrected SGD lr=1.0 + local epochs 2 + rounds 50` as the
  selected SCAFFOLD V1 setting.
- Report SCAFFOLD with communication accounting that includes control variate
  payloads.
- Compare SCAFFOLD both by rounds and by approximate communication budget,
  because its per-round payload is roughly twice FedAvg/FedProx.

Recommended next step: decide whether to run a seed sensitivity pass for the
selected FedAvg/FedProx/SCAFFOLD settings or move to the next algorithm folder.
