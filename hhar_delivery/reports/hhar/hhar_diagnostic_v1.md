# HHAR Diagnostic V1

## Purpose

This diagnostic checks the frozen HHAR V1 raw parser, synchronized split, resampling, neural model, controls, and communication accounting before algorithm tuning. It is not the final tuned FedAvg result.

## Protocol

- Seed: `20260615`; BatchNorm/Adam learning rate `0.001`.
- Centralized epochs: `15`; random-label epochs: `5`.
- Real and IID FedAvg: `10` rounds, `1` local epoch, `69` clients.
- Windows: `33040`; split windows: `{'train': 20364, 'validation': 6367, 'test': 6309}`.
- Elapsed wall time: `100.4` seconds.

## Health Gates

| Gate | Status | Observed |
| --- | --- | --- |
| manifest_integrity | PASS | {"all_retained_clients_have_train": true, "class_support_all_splits": true} |
| tiny_overfit | PASS | {"accuracy": 1.0, "macro_f1": 1.0} |
| random_label_negative_control | PASS | {"chance_scale": 0.16666666666666666, "validation_macro_f1": 0.06214958856363592} |
| centralized_learnability | PASS | {"centralized_validation_macro_f1": 0.6042259089980102, "margin": 0.5420763204343743, "random_label_validation_macro_f1": 0.06214958856363592} |
| iid_vs_real_fedavg | PASS | {"iid_validation_macro_f1": 0.520125579510967, "margin": 0.12623953109800495, "real_validation_macro_f1": 0.3938860484129621} |
| validation_test_difficulty_alignment | WARN | {"centralized_gap": 0.3169107024923302, "iid_fedavg_gap": 0.2407998785388753, "real_fedavg_gap": 0.16455493234392332} |
| device_feature_shift_present | PASS | {"device_mean_l2_distance_range": 0.44108955562114716, "max_device_mean_l2_distance": 0.6330384016036987, "min_device_mean_l2_distance": 0.19194884598255157} |

## Main Metrics

| Protocol | Val Macro-F1 | Test Macro-F1 | Communication bytes |
| --- | --- | --- | --- |
| centralized | 0.6042 | 0.9211 | 0 |
| user-device FedAvg | 0.3939 | 0.5584 | 133,782,720 |
| IID-client FedAvg | 0.5201 | 0.7609 | 133,782,720 |
| random-label centralized | 0.0621 | 0.0366 | 0 |

## FedAvg Test By Device

| Device | Windows | Supported classes | Macro-F1 |
| --- | --- | --- | --- |
| nexus4_1 | 797 | 6 | 0.5296 |
| nexus4_2 | 826 | 6 | 0.5170 |
| s3_1 | 810 | 6 | 0.5217 |
| s3_2 | 889 | 6 | 0.6488 |
| s3mini_1 | 848 | 6 | 0.6006 |
| s3mini_2 | 81 | 4 | 0.2748 |
| samsungold_1 | 1029 | 6 | 0.5172 |
| samsungold_2 | 1029 | 6 | 0.5750 |

## Diagnosis

At least one health gate is not PASS; treat the diagnostic as actionable evidence rather than a clean bill of health.

HHAR retains explicit device feature shift after train-only global channel standardization: the largest device mean L2 distance is `0.6330`. The client label distribution is also non-IID under the synchronized execution split: mean JS divergence from global train is `0.2083`.

Pair-level metrics are descriptive because 69 retained user-device clients come from only nine physical users. Global Macro-F1 and per-device behavior are the primary HHAR outcomes; this dataset is not used for independent-user fairness claims.

Validation is systematically harder than test across the centralized, real-client FedAvg, and IID-client controls. This is consistent with high execution-group variance in a nine-user dataset. The subsequent pre-registered three-seed pass confirmed stable real/IID FedAvg controls, a robust IID-over-real ranking, and non-negligible centralized validation variance; see `hhar_delivery/reports/hhar/hhar_split_seed_sensitivity_v1.md`. This original setup check predates per-experiment RNG isolation, so the split-sensitivity report is authoritative for multi-split estimates. All HHAR hyperparameter selection must aggregate validation behavior over the three frozen splits rather than rely on this single run.

## Artifacts

- Overall metrics: `outputs/hhar_diagnostic_v1/overall_metrics.csv`
- Device/user metrics: `outputs/hhar_diagnostic_v1/group_metrics.csv`
- Device feature statistics: `outputs/hhar_diagnostic_v1/device_feature_shift.csv`
- Health gates: `outputs/hhar_diagnostic_v1/health_gates.json`
