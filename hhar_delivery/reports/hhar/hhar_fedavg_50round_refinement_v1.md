# HHAR FedAvg 50-Round Validation-Only Refinement V1

## Reason

The pre-registered V1 20-round winner degraded on all three splits when extended with the same constant learning rate to 50 rounds. This refinement therefore compares the V1 pilot top three at the target 50-round budget.

- V1 round-20 mean: `0.5396`.
- V1 round-50 mean: `0.3910`.
- Mean change: `-0.1486`.
- Test metrics remain unavailable to selection.

## Fixed Protocol

- Candidates: top three V1 pilot configurations.
- Three frozen execution splits; fixed model seed `20260615`.
- 50 rounds, 1 local epoch, batch size 64, full client participation.
- Selection: aggregate validation global Macro-F1 only.
- Practical-tie tolerance: `0.005`.

## Target-Budget Results

| Selected | Config | Optimizer | LR | Val mean +/- SD | Worst split | Range | Communication |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yes | batchnorm_sgd_lr0p01 | sgd | 0.01 | 0.4767 +/- 0.0095 | 0.4672 | 0.0190 | 668,913,600 |
|  | batchnorm_adam_lr0p0003 | adam | 0.0003 | 0.4139 +/- 0.0183 | 0.3931 | 0.0343 | 668,913,600 |
|  | batchnorm_sgd_lr0p03 | sgd | 0.03 | 0.3910 +/- 0.0278 | 0.3621 | 0.0555 | 668,913,600 |

Practical-tie set: `['batchnorm_sgd_lr0p01']`. Selected high-budget config: `batchnorm_sgd_lr0p01`.

## Selected 50-Round Trajectory

| Round | Val mean +/- SD | Worst split | Communication |
| --- | --- | --- | --- |
| 0 | 0.0308 +/- 0.0001 | 0.0307 | 0 |
| 5 | 0.4706 +/- 0.0078 | 0.4618 | 66,891,360 |
| 10 | 0.4715 +/- 0.0043 | 0.4666 | 133,782,720 |
| 15 | 0.4586 +/- 0.0057 | 0.4525 | 200,674,080 |
| 20 | 0.4649 +/- 0.0073 | 0.4574 | 267,565,440 |
| 25 | 0.4706 +/- 0.0074 | 0.4628 | 334,456,800 |
| 30 | 0.4796 +/- 0.0064 | 0.4732 | 401,348,160 |
| 35 | 0.4890 +/- 0.0054 | 0.4841 | 468,239,520 |
| 40 | 0.4699 +/- 0.0090 | 0.4598 | 535,130,880 |
| 45 | 0.4904 +/- 0.0039 | 0.4870 | 602,022,240 |
| 50 | 0.4767 +/- 0.0095 | 0.4672 | 668,913,600 |

## Selected Validation By Device

These metrics are descriptive and were not used for selection.

| Device | Val mean +/- SD | Range |
| --- | --- | --- |
| nexus4_1 | 0.5208 +/- 0.0051 | 0.0102 |
| nexus4_2 | 0.4493 +/- 0.0304 | 0.0607 |
| s3_1 | 0.4639 +/- 0.0187 | 0.0370 |
| s3_2 | 0.4395 +/- 0.0089 | 0.0162 |
| s3mini_1 | 0.4803 +/- 0.0113 | 0.0213 |
| s3mini_2 | 0.3801 +/- 0.0271 | 0.0503 |
| samsungold_1 | 0.4928 +/- 0.0037 | 0.0072 |
| samsungold_2 | 0.3334 +/- 0.0040 | 0.0075 |

## Frozen FedAvg References

Communication-efficient reference:

- Config: `batchnorm_sgd_lr0p03`; rounds: `20`; local epochs: `1`.
- Validation Macro-F1: `0.5396 +/- 0.0163`.
- Communication: `267,565,440` bytes.

High-budget reference:

- Config: `batchnorm_sgd_lr0p01`; rounds: `50`; local epochs: `1`.
- Validation Macro-F1: `0.4767 +/- 0.0095`.
- Communication: `668,913,600` bytes.

The two references have different communication budgets and are not used as a direct absolute-score comparison. Subsequent algorithms must be compared at the matching 20- or 50-round budget.

All candidate histories contain only `train` and `validation` performance metrics. Model-seed sensitivity remains a separate next step.
