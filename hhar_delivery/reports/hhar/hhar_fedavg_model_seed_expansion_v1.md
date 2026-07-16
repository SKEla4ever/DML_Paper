# HHAR FedAvg Five-Model-Seed Expansion V1

## Purpose

The pre-registered 3x3 pass exceeded its model-seed SD threshold. This conditional expansion adds model seeds `20260618` and `20260619` without changing the frozen FedAvg configuration or reading test performance.

## Frozen Protocol

- Split seeds: `[20260615, 20260616, 20260617]`.
- Model seeds: `[20260615, 20260616, 20260617, 20260618, 20260619]`; full `3 x 5` crossed design.
- BatchNorm, SGD momentum `0.9`, constant learning rate `0.01`, 1 local epoch, 50 rounds, full participation.
- Primary metric: validation global Macro-F1 only.
- Stability rule: model-seed marginal sample SD <= `0.05` and range <= `0.1`.
- No further seed expansion or FedAvg retuning is allowed after this pass.

## Validation Matrix

| Split seed | Model 20260615 | Model 20260616 | Model 20260617 | Model 20260618 | Model 20260619 |
| --- | --- | --- | --- | --- | --- |
| 20260615 | 0.4768 | 0.4034 | 0.5075 | 0.4916 | 0.4569 |
| 20260616 | 0.4862 | 0.4017 | 0.4982 | 0.5039 | 0.4613 |
| 20260617 | 0.4672 | 0.3961 | 0.4862 | 0.4869 | 0.4432 |

## Model-Seed Variability

| Model seed | Role | Val mean | Across-split SD | Across-split range |
| --- | --- | --- | --- | --- |
| 20260615 | development | 0.4767 | 0.0095 | 0.0190 |
| 20260616 | confirmatory | 0.4004 | 0.0038 | 0.0072 |
| 20260617 | confirmatory | 0.4973 | 0.0107 | 0.0213 |
| 20260618 | confirmatory | 0.4941 | 0.0088 | 0.0169 |
| 20260619 | confirmatory | 0.4538 | 0.0095 | 0.0181 |

- Initial 3-seed marginal SD/range: `0.0511` / `0.0969`.
- Final 5-seed balanced mean: `0.4645`.
- Final model-seed marginal SD/range: `0.0398` / `0.0969`.
- Pre-registered stability pass: `True`.

Confirmatory estimate, excluding the development seed:

- Four-seed confirmatory mean: `0.4614`.
- Confirmatory model-seed SD: `0.0452`.
- Confirmatory minus development: `-0.0153`.

## Paired Differences

| Model seed | Split 20260615 | Split 20260616 | Split 20260617 | Mean delta |
| --- | --- | --- | --- | --- |
| 20260616 | -0.0734 | -0.0844 | -0.0711 | -0.0763 |
| 20260617 | 0.0308 | 0.0120 | 0.0190 | 0.0206 |
| 20260618 | 0.0148 | 0.0177 | 0.0197 | 0.0174 |
| 20260619 | -0.0199 | -0.0248 | -0.0240 | -0.0229 |

## Split Variability

| Split seed | Val mean | Across-model SD | Across-model range |
| --- | --- | --- | --- |
| 20260615 | 0.4672 | 0.0403 | 0.1042 |
| 20260616 | 0.4703 | 0.0416 | 0.1021 |
| 20260617 | 0.4559 | 0.0379 | 0.0908 |

Split marginal SD/range: `0.0075` / `0.0143`. Model and split uncertainty remain separate.

## Descriptive Factor Decomposition

| Source | df | SS | Share of total SS |
| --- | --- | --- | --- |
| split_seed | 2 | 0.000569 | 2.9% |
| model_seed | 4 | 0.018977 | 96.1% |
| split_by_model_seed_interaction | 8 | 0.000200 | 1.0% |

The decomposition is descriptive; one deterministic run exists per cell, so interaction is residual and no p-values are reported.

## Validation By Device

| Device | Val mean +/- model-seed SD | Range | Descriptive stable |
| --- | --- | --- | --- |
| nexus4_1 | 0.4285 +/- 0.0873 | 0.2046 | False |
| nexus4_2 | 0.4113 +/- 0.0695 | 0.1800 | False |
| s3_1 | 0.4524 +/- 0.0589 | 0.1574 | False |
| s3_2 | 0.4726 +/- 0.0652 | 0.1648 | False |
| s3mini_1 | 0.4603 +/- 0.0246 | 0.0611 | True |
| s3mini_2 | 0.3993 +/- 0.0125 | 0.0306 | True |
| samsungold_1 | 0.4641 +/- 0.0321 | 0.0824 | True |
| samsungold_2 | 0.3588 +/- 0.0599 | 0.1598 | False |

Device results are descriptive and do not gate the global decision.

## Decision

Status: `five_model_seed_characterization_complete_stable_ready_for_locked_test`.

The frozen baseline now satisfies the registered model-seed stability rule.
The observed model-seed range remains material, so all headline HHAR comparisons must retain the five-seed distribution.

The next step is a separately pre-registered, evaluation-only locked test over all 15 saved final checkpoints. No checkpoint or seed may be selected.

No pooled 15-cell SD is reported because it would conflate split and model-seed uncertainty.

## Audit

All 15 cells match the frozen split identities, runtime, configuration, checkpoint schedule, and communication budget. Metrics contain only `train` and `validation`; test performance remains locked.
