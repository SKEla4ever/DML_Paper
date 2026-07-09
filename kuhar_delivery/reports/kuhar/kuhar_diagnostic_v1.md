# KU-HAR Diagnostic V1

## Purpose

This report records setup-health diagnostics for the frozen KU-HAR V1 minimum-support pipeline. The goal is to test whether the pipeline behaves sensibly before adding more federated algorithms or moving to HHAR/WISDM.

## Protocol

- Cohort: `minimum_support`
- Seed: `20260615`
- Model family: 1D CNN with `batchnorm`
- Optimizer: `adam`, learning rate `0.001`
- Centralized oracle epochs: `30`
- Random-label negative-control epochs: `8`
- Tiny-overfit epochs: `200`
- FedAvg diagnostic budget: `20` rounds, `2` local epochs
- Local-only epochs: `20`
- Elapsed wall time: `86.2` seconds

## Manifest Snapshot

- Windows: `13627`
- Subjects: `50`
- Evaluable subjects: `50`
- Split windows: `{'train': 8177, 'validation': 2632, 'test': 2818}`
- Labels: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]`

## Health Gates

| Gate | Status | Criterion | Observed |
| --- | --- | --- | --- |
| tiny_overfit | PASS | Tiny train subset accuracy and Macro-F1 should both reach at least 0.95. | {"accuracy": 1.0, "macro_f1": 1.0} |
| random_label_negative_control | PASS | Validation Macro-F1 after permuted training labels should stay near chance; threshold=0.1200. | {"chance_macro_f1_scale": 0.058823529411764705, "validation_macro_f1": 0.010537974252655874} |
| centralized_oracle | PASS | Pooled centralized training should beat the same-budget user-split FedAvg by a visible margin. | {"centralized_validation_macro_f1": 0.8399752550826167, "margin": 0.5381482047841017, "user_fedavg_validation_macro_f1": 0.3018270502985149} |
| iid_vs_user_fedavg | PASS | IID synthetic clients should not be materially worse than user-split clients at the same FedAvg budget. | {"iid_validation_macro_f1": 0.36862613745638984, "margin": 0.06679908715787491, "user_validation_macro_f1": 0.3018270502985149} |
| client_label_heterogeneity | PASS | User clients should show visible label-distribution heterogeneity; otherwise FL algorithms may naturally look similar. | {"mean_js_divergence_from_global_train": 0.1496134154435614, "median_nonzero_train_classes_per_client": 10.0, "num_classes": 17} |

## Main Diagnostic Metrics

| Protocol | Val Macro-F1 | Val User Mean | Val Worst10 | Test Macro-F1 | Test User Mean | Test Worst10 |
| --- | --- | --- | --- | --- | --- | --- |
| centralized | 0.8400 | 0.8823 | 0.6308 | 0.8184 | 0.8869 | 0.6231 |
| user-split FedAvg | 0.3018 | 0.4745 | 0.2284 | 0.3004 | 0.4959 | 0.2419 |
| IID-client FedAvg | 0.3686 | 0.5332 | 0.3112 | 0.3866 | 0.5792 | 0.3110 |
| local-only | 0.3407 | 0.4195 | 0.1702 | 0.4484 | 0.5359 | 0.2142 |
| random-label centralized | 0.0105 | 0.0415 | 0.0000 | 0.0127 | 0.0662 | 0.0000 |

## Diagnosis

All health gates pass.

The strongest setup signal is that pooled centralized training is strong while the random-label negative control stays near chance. Centralized oracle validation Macro-F1 is `0.8400`, which is `0.5381` above user-split FedAvg at the same diagnostic comparison point. The random-label negative control stays near chance at `0.0105`.

The IID-client FedAvg control is also directionally better than the real user-split FedAvg protocol, but the gap is modest rather than oracle-like: IID-client validation Macro-F1 is `0.3686` versus `0.3018` for user-split FedAvg, a margin of `0.0668`. This points to both user heterogeneity and limited FL optimization budget as bottlenecks.

This supports the interpretation that the data loader, label mapping, model, loss, and evaluation loop are functioning. KU-HAR V1 is therefore a real non-IID challenge under this protocol, not an obviously broken setup. Similar scores among some algorithms should be treated as an algorithm/dataset interaction to analyze, not as immediate evidence that the pipeline failed.

## Client Heterogeneity

- Train clients: `50`
- Mean train windows per client: `163.5`
- Median nonzero train classes per client: `10.0`
- Mean JS divergence from global train label distribution: `0.1496`

## Existing Selected FedAvg Reference

- Run directory: `outputs/kuhar_1d_cnn_fedavg_tuning_v1/batchnorm_adam_lr0p001_e2_r50`
- Round: `50`
- Validation Macro-F1: `0.3964`
- Test Macro-F1: `0.4181`
- Communication: `499,020,000` bytes

## Artifacts

- Overall metrics: `outputs/kuhar_diagnostic_v1/overall_metrics.csv`
- Per-class metrics: `outputs/kuhar_diagnostic_v1/per_class_metrics.csv`
- Per-user metrics: `outputs/kuhar_diagnostic_v1/per_user_metrics.csv`
- Confusion matrices: `outputs/kuhar_diagnostic_v1/confusion_matrices.json`
- Client label distribution: `outputs/kuhar_diagnostic_v1/client_label_distribution.csv`
- Health gate JSON: `outputs/kuhar_diagnostic_v1/health_gates.json`

## Interpretation

KU-HAR V1 should remain in the paper as a controlled subject-heterogeneous benchmark. The next scientific step is to add cross-dataset validation on HHAR or WISDM, so the paper can distinguish KU-HAR-specific behavior from algorithm behavior that persists under stronger device or subject heterogeneity.
