# KU-HAR FedRep V1

## Purpose

This report records the first FedRep implementation and tuning pass on the
frozen KU-HAR V1 minimum-support cohort.

FedRep follows the shared-representation personalization idea from Collins et
al., "Exploiting Shared Representations for Personalized Federated Learning"
(arXiv:2102.07078, https://arxiv.org/abs/2102.07078). The paper motivates
learning a global shared representation while keeping unique local heads for
each client.

In this 1D-CNN implementation:

- Representation: every model layer before the final `Linear` classifier.
- Local head: the final `Linear` classifier.
- Communication: only representation state is uploaded/downloaded.
- Evaluation: known-client personalization, where each subject is evaluated
  with the shared representation plus that subject's train-split-updated local
  head. Validation/test data are not used for adaptation.

During local training, head updates freeze the representation and keep it in
eval mode so BatchNorm running statistics and Dropout behavior do not change
while optimizing the head. Representation updates freeze the head and train the
representation.

## Fixed Protocol

- Cohort: minimum-support
- Clients: 50
- Labels: 17 (`0`-`16`)
- Normalization: BatchNorm inside the shared representation
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Client fraction: 1.0
- Selected setting: head epochs 5, representation steps 5
- Seeds for selected setting: `20260615`, `20260616`, `20260617`

## Communication Accounting

FedRep transmits only the shared representation.

- Bytes per transmitted representation state: 95,384
- Client-local head state not transmitted: 4,420 bytes per client head
- 20 rounds with 50 clients: 190,768,000 bytes
- 50 rounds with 50 clients: 476,920,000 bytes

This is slightly lower than FedAvg/FedProx/FedBN at 50 rounds, but FedRep has
substantially higher local compute because each round performs separate head
and representation optimization phases.

## 20-Round Pilot Grid

All pilot runs use 20 rounds, Adam `lr=0.001`, BatchNorm, batch size 64, and
seed `20260615`.

| Rank | Head epochs | Representation steps | Validation Macro-F1 | Validation user mean Macro-F1 | Test Macro-F1 snapshot | Test user mean Macro-F1 snapshot | Total communication |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 5 | 0.3825 | 0.4852 | 0.5169 | 0.6499 | 190,768,000 |
| 2 | 1 | 5 | 0.2691 | 0.3904 | 0.3360 | 0.4864 | 190,768,000 |
| 3 | 5 | 1 | 0.2175 | 0.3028 | 0.2657 | 0.2928 | 190,768,000 |
| 4 | 1 | 1 | 0.1629 | 0.2027 | 0.1711 | 0.1710 | 190,768,000 |

The paper-style "many head updates, one representation update" setting
under-trained the shared representation in this CNN setting. Increasing
representation steps to 5 was necessary for competitive validation Macro-F1.

## Selected 50-Round Results

The best 20-round pilot setting, `head_epochs=5` and `representation_steps=5`,
was extended to 50 rounds and run with three seeds.

| Method | Seed | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Test Macro-F1 | Test user mean Macro-F1 | Test worst 10% user Macro-F1 | Total communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FedRep | 20260615 | 0.4704 | 0.6157 | 0.3209 | 0.6430 | 0.8190 | 0.4213 | 476,920,000 |
| FedRep | 20260616 | 0.4660 | 0.6146 | 0.3243 | 0.6325 | 0.8104 | 0.3945 | 476,920,000 |
| FedRep | 20260617 | 0.4687 | 0.6234 | 0.3209 | 0.6441 | 0.8185 | 0.4153 | 476,920,000 |

Aggregate selected-setting results:

| Metric | Mean | Std |
|---|---:|---:|
| Validation Macro-F1 | 0.4684 | 0.0022 |
| Validation user mean Macro-F1 | 0.6179 | 0.0048 |
| Validation worst 10% user Macro-F1 | 0.3221 | 0.0020 |
| Test Macro-F1 | 0.6399 | 0.0064 |
| Test user mean Macro-F1 | 0.8160 | 0.0048 |
| Test worst 10% user Macro-F1 | 0.4104 | 0.0141 |
| Validation loss | 8.2059 | 0.0359 |
| Test loss | 4.1990 | 0.0586 |

## Baseline Comparison

| Method | Personalization | Validation Macro-F1 mean | Validation user mean Macro-F1 mean | Validation worst 10% user Macro-F1 mean | Test Macro-F1 mean | Test user mean Macro-F1 mean | Total communication mean |
|---|---|---:|---:|---:|---:|---:|---:|
| SCAFFOLD | No | 0.5994 | 0.7076 | 0.4432 | 0.5922 | 0.7113 | 991,520,000 |
| FedRep | Local head | 0.4684 | 0.6179 | 0.3221 | 0.6399 | 0.8160 | 476,920,000 |
| FedAvg | No | 0.4060 | 0.5598 | 0.3166 | 0.4280 | 0.6324 | 499,020,000 |
| FedBN | Local BN | 0.4051 | 0.5570 | 0.2744 | 0.4653 | 0.6853 | 486,100,000 |
| FedProx | No | 0.4039 | 0.5572 | 0.3116 | 0.4249 | 0.6294 | 499,020,000 |

FedRep is not directly equivalent to non-personalized methods because it
evaluates known clients with local heads trained from the train split. Under
that known-client personalization protocol, it is the strongest current method
on test Macro-F1 and test per-user mean Macro-F1. SCAFFOLD remains strongest on
validation Macro-F1 and validation user-level metrics.

## Interpretation

FedRep V1 is implemented and validated. It gives a clear personalization gain
over FedAvg/FedProx/FedBN on test Macro-F1 and per-user test Macro-F1, while
using slightly less communication than FedAvg/FedProx/FedBN at 50 rounds.

The validation/test loss values are high despite strong Macro-F1. This means
the personalized heads can be over-confident or poorly calibrated. The current
claim should therefore be about classification metrics, not calibrated
probabilities.

Current conclusion for the experiment matrix:

- Include FedRep as the first strong personalization baseline.
- Report it separately from global-model methods because it uses known-client
  local heads.
- Do not claim it beats SCAFFOLD overall: SCAFFOLD is better on validation
  Macro-F1, while FedRep is better on test Macro-F1 and test per-user mean.
- Track local compute cost, because FedRep trades communication for many local
  head/representation updates.

Recommended next step: implement Ditto as a second personalization baseline,
because Ditto personalizes the full model with regularization rather than only
the classifier head.
