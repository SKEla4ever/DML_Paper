# KU-HAR FedBN V1

## Purpose

This report records the first FedBN implementation and seed-sensitivity pass on
the frozen KU-HAR V1 minimum-support cohort.

FedBN keeps the same 1D-CNN backbone, frozen manifest loader, and metrics as
the tuned FedAvg baseline. The algorithmic change is that BatchNorm state is
client-local:

- BatchNorm affine parameters and running statistics are not uploaded.
- Non-BatchNorm model state is aggregated with FedAvg.
- Evaluation is client-specific: each subject is evaluated with the shared
  global Conv/Linear weights plus that subject's local BatchNorm state.

Selection/comparison uses validation Macro-F1 as the primary metric. Test
metrics are recorded as snapshots for audit visibility.

## Fixed Setting

- Cohort: minimum-support
- Clients: 50
- Labels: 17 (`0`-`16`)
- Normalization: BatchNorm, client-local
- Optimizer: Adam
- Learning rate: 0.001
- Local epochs: 2
- Batch size: 64
- Client fraction: 1.0
- Rounds: 50
- Seeds: `20260615`, `20260616`, `20260617`

## Communication Accounting

FedBN transmits only the shared non-BatchNorm state.

- Bytes per transmitted shared state: 97,220
- Client-local BatchNorm state not transmitted: 2,584 bytes per client state
- 50 rounds with 50 clients: 486,100,000 bytes

This is slightly lower than FedAvg/FedProx at the same number of rounds because
BatchNorm state is excluded from communication.

## Per-Seed Results

| Method | Seed | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Test Macro-F1 | Test user mean Macro-F1 | Total communication |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedBN | 20260615 | 0.4081 | 0.5672 | 0.2957 | 0.4677 | 0.6956 | 486,100,000 |
| FedBN | 20260616 | 0.3952 | 0.5620 | 0.2594 | 0.4475 | 0.6815 | 486,100,000 |
| FedBN | 20260617 | 0.4120 | 0.5419 | 0.2680 | 0.4807 | 0.6787 | 486,100,000 |

## Aggregate Comparison

| Method | Validation Macro-F1 mean | Validation Macro-F1 std | Validation user mean Macro-F1 mean | Validation worst 10% user Macro-F1 mean | Test Macro-F1 mean | Test Macro-F1 std | Test user mean Macro-F1 mean | Total communication mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAFFOLD | 0.5994 | 0.0067 | 0.7076 | 0.4432 | 0.5922 | 0.0028 | 0.7113 | 991,520,000 |
| FedAvg | 0.4060 | 0.0171 | 0.5598 | 0.3166 | 0.4280 | 0.0090 | 0.6324 | 499,020,000 |
| FedBN | 0.4051 | 0.0088 | 0.5570 | 0.2744 | 0.4653 | 0.0167 | 0.6853 | 486,100,000 |
| FedProx | 0.4039 | 0.0147 | 0.5572 | 0.3116 | 0.4249 | 0.0103 | 0.6294 | 499,020,000 |

## Trajectory Snapshot

FedBN validation Macro-F1 at 50 rounds:

| Seed | Round 10 | Round 20 | Round 30 | Round 40 | Round 50 |
|---:|---:|---:|---:|---:|---:|
| 20260615 | 0.2514 | 0.2990 | 0.3389 | 0.3672 | 0.4081 |
| 20260616 | 0.3046 | 0.3341 | 0.3831 | 0.3693 | 0.3952 |
| 20260617 | 0.2980 | 0.3546 | 0.3826 | 0.3731 | 0.4120 |

## Interpretation

FedBN is implemented and validated, but it does not clearly improve validation
Macro-F1 over FedAvg/FedProx in this first pass. The three-seed validation mean
is effectively tied with FedAvg:

- FedAvg validation Macro-F1 mean: 0.4060
- FedBN validation Macro-F1 mean: 0.4051
- FedProx validation Macro-F1 mean: 0.4039

FedBN does improve the held-out test snapshot relative to FedAvg/FedProx:

- FedBN test Macro-F1 mean: 0.4653
- FedAvg test Macro-F1 mean: 0.4280
- FedProx test Macro-F1 mean: 0.4249

This suggests that client-local BatchNorm captures some subject-specific
feature distribution shift, but the validation-selected conclusion should be
conservative: FedBN is a useful lightweight heterogeneity baseline, not a new
best method. SCAFFOLD remains the strongest current baseline by a large margin.

Current conclusion for the experiment matrix:

- Include FedBN as an ablation for local normalization/statistics.
- Do not replace SCAFFOLD or tuned FedAvg with FedBN as the selected main
  baseline.
- Mention the validation/test split tension: validation Macro-F1 is tied with
  FedAvg/FedProx, while test Macro-F1 improves.

Recommended next step: move to a personalization baseline, either FedRep or
Ditto, because FedBN only localizes normalization while keeping a global
classifier.
