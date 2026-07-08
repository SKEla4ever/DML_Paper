# KU-HAR Ditto V1

## Purpose

This report records the first Ditto implementation and tuning pass on the
frozen KU-HAR V1 minimum-support cohort.

Ditto follows the personalization objective from Li et al., "Ditto: Fair and
Robust Federated Learning Through Personalization" (arXiv:2012.04221,
https://arxiv.org/abs/2012.04221). Each client maintains a full personalized
model `v_k` regularized toward the current global model `w`:

`cross_entropy(v_k) + lambda / 2 * ||v_k - w||^2`

In this implementation:

- Global model: trained with FedAvg-style aggregation.
- Personalized model: full client-local 1D-CNN, updated from the client's
  previous personalized model.
- Communication: only global model updates are uploaded/downloaded.
- Evaluation: known-client personalization, where each subject is evaluated
  with that subject's train-split-updated personalized model. Validation/test
  data are not used for adaptation.

## Fixed Protocol

- Cohort: minimum-support
- Clients: 50
- Labels: 17 (`0`-`16`)
- Normalization: BatchNorm
- Optimizer: Adam
- Learning rate: 0.001
- Global local epochs: 2
- Personalized local epochs: 2
- Batch size: 64
- Client fraction: 1.0
- Selected lambda: 0.01
- Seeds for selected setting: `20260615`, `20260616`, `20260617`

## Communication Accounting

Ditto transmits only the global model state.

- Bytes per transmitted global model state: 99,804
- Personalized model state stays local: 99,804 bytes per client state
- 20 rounds with 50 clients: 199,608,000 bytes
- 50 rounds with 50 clients: 499,020,000 bytes

Ditto has the same communication accounting as FedAvg/FedProx at the same
number of rounds, but extra local compute because each round updates both a
global branch and a personalized branch.

## 20-Round Lambda Grid

All grid runs use 20 rounds, Adam `lr=0.001`, BatchNorm, global local epochs 2,
personal local epochs 2, batch size 64, and seed `20260615`.

| Rank | Lambda | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Test Macro-F1 snapshot | Test user mean Macro-F1 snapshot | Total communication |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 0.4023 | 0.4952 | 0.2355 | 0.5714 | 0.7136 | 199,608,000 |
| 2 | 0.0 | 0.4019 | 0.4959 | 0.2323 | 0.5834 | 0.7188 | 199,608,000 |
| 3 | 0.1 | 0.3857 | 0.4837 | 0.2302 | 0.5323 | 0.6898 | 199,608,000 |
| 4 | 1.0 | 0.2843 | 0.4178 | 0.2048 | 0.3938 | 0.5972 | 199,608,000 |

The validation-selected setting is `lambda=0.01`, narrowly ahead of the
local-only `lambda=0.0` control. Stronger regularization degraded both
validation and test Macro-F1.

## Selected 50-Round Results

The best 20-round lambda by validation Macro-F1, `lambda=0.01`, was extended to
50 rounds and run with three seeds.

| Method | Seed | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Test Macro-F1 | Test user mean Macro-F1 | Test worst 10% user Macro-F1 | Total communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ditto | 20260615 | 0.4643 | 0.5988 | 0.3003 | 0.6505 | 0.8042 | 0.4035 | 499,020,000 |
| Ditto | 20260616 | 0.4668 | 0.6194 | 0.3429 | 0.6392 | 0.7918 | 0.3563 | 499,020,000 |
| Ditto | 20260617 | 0.4677 | 0.6104 | 0.3261 | 0.6364 | 0.8014 | 0.3741 | 499,020,000 |

Aggregate selected-setting results:

| Metric | Mean | Std |
|---|---:|---:|
| Validation Macro-F1 | 0.4663 | 0.0018 |
| Validation user mean Macro-F1 | 0.6095 | 0.0103 |
| Validation worst 10% user Macro-F1 | 0.3231 | 0.0214 |
| Test Macro-F1 | 0.6420 | 0.0074 |
| Test user mean Macro-F1 | 0.7991 | 0.0065 |
| Test worst 10% user Macro-F1 | 0.3780 | 0.0238 |
| Validation loss | 3.3047 | 0.0347 |
| Test loss | 1.8555 | 0.0162 |

## Baseline Comparison

| Method | Personalization | Validation Macro-F1 mean | Validation user mean Macro-F1 mean | Validation worst 10% user Macro-F1 mean | Test Macro-F1 mean | Test user mean Macro-F1 mean | Test loss mean | Total communication mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SCAFFOLD | No | 0.5994 | 0.7076 | 0.4432 | 0.5922 | 0.7113 | not summarized | 991,520,000 |
| FedRep | Local head | 0.4684 | 0.6179 | 0.3221 | 0.6399 | 0.8160 | 4.1990 | 476,920,000 |
| Ditto | Full local model | 0.4663 | 0.6095 | 0.3231 | 0.6420 | 0.7991 | 1.8555 | 499,020,000 |
| FedAvg | No | 0.4060 | 0.5598 | 0.3166 | 0.4280 | 0.6324 | not summarized | 499,020,000 |
| FedBN | Local BN | 0.4051 | 0.5570 | 0.2744 | 0.4653 | 0.6853 | not summarized | 486,100,000 |
| FedProx | No | 0.4039 | 0.5572 | 0.3116 | 0.4249 | 0.6294 | not summarized | 499,020,000 |

Ditto is not directly equivalent to global-model methods because it evaluates
known clients with full personalized models trained on each client's train
split. Under that known-client personalization protocol, Ditto and FedRep are
the strongest current methods for test Macro-F1. SCAFFOLD remains the strongest
method by validation Macro-F1 and validation user-level metrics.

## Interpretation

Ditto V1 is implemented and validated. It provides a strong full-model
personalization baseline:

- Validation Macro-F1 is essentially tied with FedRep.
- Test Macro-F1 is slightly higher than FedRep.
- Test loss is much lower than FedRep, suggesting less extreme
  over-confidence.
- Communication matches FedAvg/FedProx at the same number of rounds, but local
  compute is higher because Ditto trains global and personalized branches.

Current conclusion for the experiment matrix:

- Include Ditto as the main full-model personalization baseline.
- Report Ditto separately from global-model methods because it uses
  known-client personalized models.
- Treat Ditto and FedRep as complementary: FedRep has slightly better test
  user mean Macro-F1, while Ditto has slightly better test Macro-F1 and much
  lower loss.
- SCAFFOLD remains the strongest non-personalized method and the strongest
  method by validation Macro-F1.

Recommended next step: decide whether the paper's main story should compare
global methods and personalization methods in separate tables, then proceed to
FedProto or communication/compression variants.
