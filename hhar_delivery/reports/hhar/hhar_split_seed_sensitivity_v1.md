# HHAR Split-Seed Sensitivity V1

## Purpose

This pass isolates execution-split variance before any HHAR hyperparameter selection. Only the split seed changes. Model initialization, optimizer seed, architecture, training budget, client count, and communication accounting remain fixed.

## Pre-Registered Protocol

- Split seeds: `[20260615, 20260616, 20260617]`
- Fixed model/optimizer seed: `20260615`
- Stability threshold: sample Macro-F1 standard deviation <= `0.05` and range <= `0.10` for both validation and test.
- Centralized: `15` epochs.
- FedAvg controls: `10` rounds x `1` local epoch; real and IID each use `69` clients.
- Runtime check: every completed diagnostic used the same resolved device, `mps`.
- Test metrics diagnose split difficulty only; they are not used for hyperparameter selection.

## Frozen Split Identities

| Split seed | Train windows | Val windows | Test windows | Resolved device | Execution manifest SHA-256 | Window manifest SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| 20260615 | 20364 | 6367 | 6309 | mps | 9e510385b4f222171729d75c76ca531a596e8452c99c9fb08e97d574a0a7fa48 | 194b85c6946f4fd5aaa83bf9142a16a28a373754a505ea32c7f83fbe68f45962 |
| 20260616 | 20418 | 6313 | 6309 | mps | 1737fd800727d49841089977adfb5576e145e2ec698caf8a820aba2ef3adcb29 | a9e4a03e15f9e448c979e3fe748e21a1f6a7a8081d00d6a68c607cf746f0997c |
| 20260617 | 20382 | 6364 | 6294 | mps | 26866698b16d31e55e39faf5b345718d2c045441b1401655d2e6374937bcc2c4 | 5b7c9634be850864a47add60e7bf01b4b12eeca53f8f3a8d2d6a9a8cb638987e |

## Setup Controls

| Split seed | Tiny-overfit Macro-F1 | Random-label Val Macro-F1 |
| --- | --- | --- |
| 20260615 | 1.0000 | 0.0724 |
| 20260616 | 1.0000 | 0.1088 |
| 20260617 | 1.0000 | 0.1700 |

## Per-Seed Results

| Split seed | Protocol | Val Macro-F1 | Test Macro-F1 | Test - Val | Communication |
| --- | --- | --- | --- | --- | --- |
| 20260615 | centralized | 0.5378 | 0.8858 | 0.3480 | 0 |
| 20260615 | user-device FedAvg | 0.3939 | 0.5584 | 0.1646 | 133,782,720 |
| 20260615 | IID-client FedAvg | 0.5201 | 0.7609 | 0.2408 | 133,782,720 |
| 20260616 | centralized | 0.6351 | 0.8837 | 0.2486 | 0 |
| 20260616 | user-device FedAvg | 0.3915 | 0.5773 | 0.1858 | 133,782,720 |
| 20260616 | IID-client FedAvg | 0.5340 | 0.7678 | 0.2338 | 133,782,720 |
| 20260617 | centralized | 0.5321 | 0.8419 | 0.3098 | 0 |
| 20260617 | user-device FedAvg | 0.3784 | 0.5865 | 0.2082 | 133,782,720 |
| 20260617 | IID-client FedAvg | 0.5195 | 0.7755 | 0.2560 | 133,782,720 |

## Aggregate Split Variance

| Protocol | Val mean +/- SD | Val range | Test mean +/- SD | Test range | Stable |
| --- | --- | --- | --- | --- | --- |
| centralized | 0.5683 +/- 0.0579 | 0.1030 | 0.8705 +/- 0.0248 | 0.0439 | False |
| user-device FedAvg | 0.3879 +/- 0.0084 | 0.0155 | 0.5741 +/- 0.0143 | 0.0281 | True |
| IID-client FedAvg | 0.5245 +/- 0.0082 | 0.0145 | 0.7681 +/- 0.0073 | 0.0146 | True |

## Split Assignment Overlap

| Seed A | Seed B | Same-split fraction | Val Jaccard | Test Jaccard |
| --- | --- | --- | --- | --- |
| 20260615 | 20260616 | 0.8077 | 0.5526 | 0.5385 |
| 20260615 | 20260617 | 0.8077 | 0.5676 | 0.5641 |
| 20260616 | 20260617 | 0.8077 | 0.5946 | 0.6053 |

## FedAvg Test Performance By Device

| Device | Mean +/- SD | Range | Stable |
| --- | --- | --- | --- |
| nexus4_1 | 0.5423 +/- 0.0130 | 0.0261 | True |
| nexus4_2 | 0.5368 +/- 0.0185 | 0.0366 | True |
| s3_1 | 0.5286 +/- 0.0074 | 0.0147 | True |
| s3_2 | 0.6643 +/- 0.0144 | 0.0284 | True |
| s3mini_1 | 0.6207 +/- 0.0189 | 0.0374 | True |
| s3mini_2 | 0.3467 +/- 0.0661 | 0.1300 | False |
| samsungold_1 | 0.5295 +/- 0.0108 | 0.0199 | True |
| samsungold_2 | 0.5925 +/- 0.0155 | 0.0292 | True |

## IID Minus Real FedAvg

| Split seed | Validation margin | Test margin |
| --- | --- | --- |
| 20260615 | 0.1262 | 0.2025 |
| 20260616 | 0.1425 | 0.1905 |
| 20260617 | 0.1412 | 0.1890 |

## Interpretation

Both FedAvg controls satisfy the pre-registered split-stability thresholds, but the centralized validation oracle does not. The IID-versus-real FedAvg ranking is robust, while centralized absolute validation performance and the weak-device estimate require multi-split reporting.

Setup controls pass on all split seeds: `True`. Both FL controls pass the stability thresholds: `True`; all three protocols pass: `False`. IID-client FedAvg ranks above real user-device FedAvg on validation and test for every split seed: `True`. Ranking consistency and absolute-score stability are reported separately; a stable ranking does not make a noisy single-split score precise.

The test-minus-validation offset remains positive for every protocol and split seed. Mean offsets are centralized `0.3021`, real FedAvg `0.1862`, and IID FedAvg `0.2435`. This is a systematic split-difficulty difference, not an isolated seed failure.

Seven of eight device-level FedAvg test summaries pass the thresholds. `s3mini_2` remains unstable because it is the pre-identified weak-device condition with very limited supported windows; it must be reported separately from the seven adequately supported devices.

Because HHAR contains only nine physical users, this sensitivity is treated as grouped execution-split uncertainty. User-device pairs remain repeated measurements and are not counted as independent statistical units.

## Artifacts

- Per-seed protocol results: `outputs/hhar_split_seed_sensitivity_v1/split_results.csv`
- Protocol summary: `outputs/hhar_split_seed_sensitivity_v1/protocol_summary.csv`
- Device results: `outputs/hhar_split_seed_sensitivity_v1/device_results.csv`
- Device summary: `outputs/hhar_split_seed_sensitivity_v1/device_summary.csv`
- Split overlap: `outputs/hhar_split_seed_sensitivity_v1/split_overlap.csv`
- Pre-registration: `outputs/hhar_split_seed_sensitivity_v1/pre_registration.json`
- Published machine-readable summary: `hhar_delivery/reports/hhar/hhar_split_seed_sensitivity_v1_summary.json`
