# HHAR FedAvg Locked Test V1

## Boundary

This is the first and final test evaluation of the frozen HHAR FedAvg baseline. All 15 pre-registered checkpoints were loaded without retraining or checkpoint selection. Test results were not used for tuning.

## Global Test Results

| Split seed | Model 20260615 | Model 20260616 | Model 20260617 | Model 20260618 | Model 20260619 |
| --- | --- | --- | --- | --- | --- |
| 20260615 | 0.6050 | 0.5965 | 0.6181 | 0.6163 | 0.6424 |
| 20260616 | 0.6223 | 0.6199 | 0.6412 | 0.6330 | 0.6658 |
| 20260617 | 0.6411 | 0.6348 | 0.6504 | 0.6507 | 0.6825 |

| Model seed | Role | Test Macro-F1 mean +/- split SD | Test Accuracy | Validation Macro-F1 | Test - validation |
| --- | --- | --- | --- | --- | --- |
| 20260615 | development | 0.6228 +/- 0.0180 | 0.6180 | 0.4767 | 0.1461 |
| 20260616 | confirmatory | 0.6171 +/- 0.0193 | 0.6123 | 0.4004 | 0.2167 |
| 20260617 | confirmatory | 0.6365 +/- 0.0166 | 0.6316 | 0.4973 | 0.1392 |
| 20260618 | confirmatory | 0.6333 +/- 0.0172 | 0.6252 | 0.4941 | 0.1392 |
| 20260619 | confirmatory | 0.6636 +/- 0.0201 | 0.6541 | 0.4538 | 0.2097 |

- Balanced test Macro-F1: `0.6347`.
- Balanced test Accuracy: `0.6282`.
- Model-seed marginal SD/range: `0.0180` / `0.0465`.
- Confirmatory four-seed test Macro-F1: `0.6376 +/- 0.0193`.
- Confirmatory minus development: `0.0148`.
- Balanced test-minus-validation Macro-F1: `0.1702`.

## Split Variability

| Split seed | Test mean +/- model SD | Range | Test - validation |
| --- | --- | --- | --- |
| 20260615 | 0.6157 +/- 0.0173 | 0.0459 | 0.1484 |
| 20260616 | 0.6364 +/- 0.0185 | 0.0459 | 0.1662 |
| 20260617 | 0.6519 +/- 0.0183 | 0.0476 | 0.1959 |

Split-seed marginal SD/range: `0.0182` / `0.0362`. This remains separate from model-seed uncertainty.

## Descriptive Factor Decomposition

| Source | df | SS | Share of total SS |
| --- | --- | --- | --- |
| split_seed | 2 | 0.003305 | 45.8% |
| model_seed | 4 | 0.003869 | 53.6% |
| split_by_model_seed_interaction | 8 | 0.000048 | 0.7% |

This decomposition is descriptive; no p-values are reported.

## Per Activity

| Activity | F1 mean +/- model SD | Precision | Recall | F1 range | Mean test windows/split |
| --- | --- | --- | --- | --- | --- |
| bike | 0.5353 +/- 0.0244 | 0.4902 | 0.5913 | 0.0557 | 1072.0 |
| sit | 0.7964 +/- 0.0161 | 0.6620 | 1.0000 | 0.0378 | 667.0 |
| stand | 0.6169 +/- 0.0794 | 0.9726 | 0.4554 | 0.1843 | 1162.0 |
| walk | 0.6916 +/- 0.0046 | 0.9480 | 0.5445 | 0.0108 | 1364.0 |
| stairsup | 0.5973 +/- 0.0083 | 0.4472 | 0.8993 | 0.0199 | 1064.0 |
| stairsdown | 0.5705 +/- 0.0318 | 0.8132 | 0.4419 | 0.0753 | 975.0 |

## Per Device

| Device | Macro-F1 mean +/- model SD | Range | Supported classes | Mean windows/cell |
| --- | --- | --- | --- | --- |
| nexus4_1 | 0.5905 +/- 0.0318 | 0.0750 | 6 | 796.3 |
| nexus4_2 | 0.6531 +/- 0.0603 | 0.1367 | 6 | 827.0 |
| s3_1 | 0.6080 +/- 0.0095 | 0.0212 | 6 | 808.0 |
| s3_2 | 0.6881 +/- 0.0081 | 0.0213 | 6 | 888.0 |
| s3mini_1 | 0.6569 +/- 0.0281 | 0.0699 | 6 | 847.3 |
| s3mini_2 | 0.3613 +/- 0.0014 | 0.0026 | 4 | 80.7 |
| samsungold_1 | 0.6334 +/- 0.0328 | 0.0751 | 6 | 1028.3 |
| samsungold_2 | 0.6516 +/- 0.0158 | 0.0338 | 6 | 1028.3 |

Device values use supported-class Macro-F1 and are descriptive.

## Per Physical User

| Physical user | Macro-F1 mean +/- model SD | Range | Supported classes | Mean windows/cell |
| --- | --- | --- | --- | --- |
| a | 0.0242 +/- 0.0196 | 0.0437 | 2 | 704.0 |
| b | 0.7391 +/- 0.0250 | 0.0652 | 2-3 | 775.0 |
| c | 0.1963 +/- 0.0449 | 0.1108 | 2 | 642.3 |
| d | 0.9503 +/- 0.0156 | 0.0392 | 2 | 689.7 |
| e | 0.6909 +/- 0.0486 | 0.1094 | 2 | 763.7 |
| f | 0.8063 +/- 0.0457 | 0.1115 | 2 | 653.7 |
| g | 0.6251 +/- 0.0330 | 0.0753 | 3 | 682.0 |
| h | 0.4869 +/- 0.0724 | 0.1601 | 2 | 645.7 |
| i | 0.9940 +/- 0.0067 | 0.0166 | 1 | 748.0 |

Mean physical-user Macro-F1: `0.6126`; user-level bootstrap 95% CI `[0.4029, 0.8033]` over `9` physical users.

Physical-user test support ranges from one to three activities per cell. These supported-class scores and their bootstrap interval are descriptive; they are not a six-class fairness estimate.

User-device pairs are retained only as descriptive client-level metrics; they are not treated as independent users.
Their balanced mean Macro-F1 is `0.6619` and the balanced worst-10% value is `0.4776`.

## Mean Normalized Confusion

| True / predicted | bike | sit | stand | walk | stairsup | stairsdown |
| --- | --- | --- | --- | --- | --- | --- |
| bike | 0.5913 | 0.1235 | 0.0099 | 0.0000 | 0.2753 | 0.0000 |
| sit | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stand | 0.5446 | 0.0000 | 0.4554 | 0.0000 | 0.0000 | 0.0000 |
| walk | 0.0033 | 0.1106 | 0.0000 | 0.5445 | 0.3241 | 0.0176 |
| stairsup | 0.0045 | 0.0000 | 0.0022 | 0.0256 | 0.8993 | 0.0684 |
| stairsdown | 0.0242 | 0.0600 | 0.0014 | 0.0139 | 0.4587 | 0.4419 |

Rows are normalized within each checkpoint and then averaged equally over all 15 cells.

## Audit

Every checkpoint SHA-256, test-window count, prediction row, class support, confusion total, device set, and physical-user set was validated. All evaluations were inference-only on `mps`; communication remains the frozen training cost of `668,913,600` bytes per checkpoint.

The locked test is now closed. No post-test FedAvg retuning or seed selection is permitted.
