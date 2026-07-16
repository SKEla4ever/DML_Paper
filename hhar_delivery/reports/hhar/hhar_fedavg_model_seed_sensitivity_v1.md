# HHAR FedAvg Model-Seed Sensitivity V1

## Purpose

This validation-only pass measures optimization-seed sensitivity after the HHAR FedAvg configuration was frozen. It does not select or retune hyperparameters. Model seed `20260615` is the development seed used during tuning; seeds `20260616` and `20260617` are confirmatory.

## Pre-Registered Protocol

- Split seeds: `[20260615, 20260616, 20260617]`.
- Model seeds: `[20260615, 20260616, 20260617]`; full `3 x 3` crossed design.
- Frozen setting: BatchNorm, SGD momentum `0.9`, constant learning rate `0.01`, 1 local epoch, 50 rounds, full participation.
- Primary metric: validation global Macro-F1; test metrics were not generated.
- Stability rule: sample SD across model-seed marginal means <= `0.05` and range <= `0.1`.
- The two confirmatory model seeds are summarized separately because the development seed contributed to hyperparameter selection.

## Crossed Results

| Split seed | Model 20260615 | Model 20260616 | Model 20260617 |
| --- | --- | --- | --- |
| 20260615 | 0.4768 | 0.4034 | 0.5075 |
| 20260616 | 0.4862 | 0.4017 | 0.4982 |
| 20260617 | 0.4672 | 0.3961 | 0.4862 |

## Model-Seed Variability

| Model seed | Role | Val mean | Across-split SD | Across-split range |
| --- | --- | --- | --- | --- |
| 20260615 | development | 0.4767 | 0.0095 | 0.0190 |
| 20260616 | confirmatory | 0.4004 | 0.0038 | 0.0072 |
| 20260617 | confirmatory | 0.4973 | 0.0107 | 0.0213 |

Across the three model-seed marginal means:

- Balanced grand mean: `0.4581`.
- Model-seed marginal sample SD: `0.0511`.
- Model-seed marginal range: `0.0969`.
- Pre-registered stability pass: `False`.

Confirmatory estimate:

- Development-seed marginal mean: `0.4767`.
- Held-out model-seed marginal mean: `0.4489`.
- Confirmatory minus development: `-0.0279`.

## Paired Seed Differences

Each difference uses the same split under the development model seed as its reference.

| Confirmatory model seed | Split 20260615 | Split 20260616 | Split 20260617 | Mean delta |
| --- | --- | --- | --- | --- |
| 20260616 | -0.0734 | -0.0844 | -0.0711 | -0.0763 |
| 20260617 | 0.0308 | 0.0120 | 0.0190 | 0.0206 |

## Split Variability

| Split seed | Val mean | Across-model SD | Across-model range |
| --- | --- | --- | --- |
| 20260615 | 0.4626 | 0.0535 | 0.1042 |
| 20260616 | 0.4620 | 0.0526 | 0.0964 |
| 20260617 | 0.4499 | 0.0475 | 0.0901 |

Split-seed marginal sample SD: `0.0072`; range: `0.0127`. These are reported separately from model-seed variability.

## Descriptive Factor Decomposition

| Source | df | SS | MS | Share of total SS |
| --- | --- | --- | --- | --- |
| split_seed | 2 | 0.000309 | 0.000155 | 1.9% |
| model_seed | 2 | 0.015639 | 0.007820 | 97.3% |
| split_by_model_seed_interaction | 4 | 0.000127 | 0.000032 | 0.8% |

This balanced two-way decomposition is descriptive. With one deterministic run per cell, the residual is the split-by-model-seed interaction; no p-values or formal variance claims are made.

## Validation By Device

Each device value first averages the three splits within each model seed; the displayed SD and range are then computed over the three model-seed marginals. Device stability is descriptive and does not gate the global decision.

| Device | Val mean +/- model-seed SD | Range | Descriptive stable |
| --- | --- | --- | --- |
| nexus4_1 | 0.4136 +/- 0.1027 | 0.2046 | False |
| nexus4_2 | 0.3934 +/- 0.0630 | 0.1241 | False |
| s3_1 | 0.4597 +/- 0.0788 | 0.1574 | False |
| s3_2 | 0.4768 +/- 0.0885 | 0.1648 | False |
| s3mini_1 | 0.4658 +/- 0.0330 | 0.0611 | True |
| s3mini_2 | 0.3938 +/- 0.0138 | 0.0275 | True |
| samsungold_1 | 0.4553 +/- 0.0417 | 0.0824 | True |
| samsungold_2 | 0.3542 +/- 0.0819 | 0.1598 | False |

## Decision

Status: `model_seed_instability_requires_five_seed_expansion`.

The frozen baseline failed the pre-registered stability rule. Hyperparameters remain frozen, test remains locked, and model-seed characterization must expand to five seeds.

No naive pooled 9-cell standard deviation is reported, because it would conflate split and model-seed uncertainty.

## Audit

All nine runs were validated against the frozen split identities, runtime, configuration, checkpoint schedule, and communication budget. Every metric artifact contains only `train` and `validation` performance.

Pre-registration source: `2fa0497c2b68983df69ed57a662589cfc20ef3ff`.
