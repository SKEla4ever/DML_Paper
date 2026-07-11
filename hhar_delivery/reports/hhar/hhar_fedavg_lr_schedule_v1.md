# HHAR FedAvg Validation-Only LR-Schedule Selection V1

## Purpose

This final bounded search checks whether a standard learning-rate decay produces a stronger 50-round FedAvg baseline than the selected constant-LR reference. It uses the same three frozen splits and no test metrics.

## Frozen Candidate Set

- Constant `0.01` reference from the target-budget refinement.
- Step decay: `0.03` through round 20, then `0.003`.
- Cosine decay: `0.03` to `0.001` over 50 rounds.
- BatchNorm, SGD momentum `0.9`, 1 local epoch, full participation.
- Stop rule: no additional HHAR FedAvg optimization candidates after this comparison.

## Results

| Selected | LR schedule | Val mean +/- SD | Worst split | Range | Communication |
| --- | --- | --- | --- | --- | --- |
| yes | constant 0.01 | 0.4767 +/- 0.0095 | 0.4672 | 0.0190 | 668,913,600 |
|  | cosine 0.03 to 0.001 over 50 rounds | 0.4567 +/- 0.0140 | 0.4425 | 0.0280 | 668,913,600 |
|  | 0.03, then 0.003 after round 20 | 0.4472 +/- 0.0131 | 0.4334 | 0.0260 | 668,913,600 |

Practical-tie set: `['batchnorm_sgd_lr0p01_constant']`. Frozen 50-round FedAvg config: `batchnorm_sgd_lr0p01_constant`.

## Selected Trajectory

| Round | Effective LR | Val mean +/- SD | Worst split | Communication |
| --- | --- | --- | --- | --- |
| 0 | 0.010000 | 0.0308 +/- 0.0001 | 0.0307 | 0 |
| 5 | 0.010000 | 0.4706 +/- 0.0078 | 0.4618 | 66,891,360 |
| 10 | 0.010000 | 0.4715 +/- 0.0043 | 0.4666 | 133,782,720 |
| 15 | 0.010000 | 0.4586 +/- 0.0057 | 0.4525 | 200,674,080 |
| 20 | 0.010000 | 0.4649 +/- 0.0073 | 0.4574 | 267,565,440 |
| 25 | 0.010000 | 0.4706 +/- 0.0074 | 0.4628 | 334,456,800 |
| 30 | 0.010000 | 0.4796 +/- 0.0064 | 0.4732 | 401,348,160 |
| 35 | 0.010000 | 0.4890 +/- 0.0054 | 0.4841 | 468,239,520 |
| 40 | 0.010000 | 0.4699 +/- 0.0090 | 0.4598 | 535,130,880 |
| 45 | 0.010000 | 0.4904 +/- 0.0039 | 0.4870 | 602,022,240 |
| 50 | 0.010000 | 0.4767 +/- 0.0095 | 0.4672 | 668,913,600 |

## Validation By Device

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

## Frozen Baseline

Primary high-budget HHAR FedAvg reference:

- Config: `batchnorm_sgd_lr0p01_constant`.
- Rounds: `50`; local epochs: `1`; full participation.
- Validation Macro-F1: `0.4767 +/- 0.0095`.
- Communication: `668,913,600` bytes.

Communication-efficient supplementary reference:

- Config: `batchnorm_sgd_lr0p03` at 20 rounds.
- Validation Macro-F1: `0.5396 +/- 0.0163`.
- Communication: `267,565,440` bytes.

The two budgets remain separate. Subsequent methods must use the matching round and communication budget. All tuning histories contain only `train` and `validation` performance metrics.

The hyperparameters and budgets are now frozen, but the reported validation uncertainty holds model seed `20260615` fixed. Run model-seed sensitivity separately before the first locked test evaluation.
