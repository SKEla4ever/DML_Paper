# KU-HAR Seed Sensitivity V1

## Purpose

This report records a first seed-sensitivity pass for the selected KU-HAR
minimum-support baselines:

- FedAvg: `BatchNorm + Adam lr=0.001 + local epochs 2 + rounds 50`
- FedProx: `BatchNorm + Adam lr=0.001 + mu=0.1 + local epochs 2 + rounds 50`
- SCAFFOLD: `BatchNorm + corrected SGD lr=1.0 + local epochs 2 + rounds 50`

The sweep uses seeds `20260615`, `20260616`, and `20260617`. Existing selected
seed `20260615` runs were reused; missing runs were generated with
`kuhar_delivery/scripts/run_kuhar_seed_sensitivity.py`.

All reported standard deviations are sample standard deviations over the three
seeds. This is a stability check, not a formal significance test.

## Per-Seed Results

| Method | Seed | Validation Macro-F1 | Validation user mean Macro-F1 | Validation worst 10% user Macro-F1 | Test Macro-F1 | Test user mean Macro-F1 | Total communication |
|---|---:|---:|---:|---:|---:|---:|---:|
| FedAvg | 20260615 | 0.3964 | 0.5567 | 0.3211 | 0.4181 | 0.6289 | 499,020,000 |
| FedAvg | 20260616 | 0.4257 | 0.5838 | 0.3124 | 0.4303 | 0.6496 | 499,020,000 |
| FedAvg | 20260617 | 0.3959 | 0.5388 | 0.3163 | 0.4356 | 0.6188 | 499,020,000 |
| FedProx | 20260615 | 0.3970 | 0.5554 | 0.3053 | 0.4158 | 0.6269 | 499,020,000 |
| FedProx | 20260616 | 0.4207 | 0.5787 | 0.3133 | 0.4228 | 0.6430 | 499,020,000 |
| FedProx | 20260617 | 0.3939 | 0.5376 | 0.3163 | 0.4361 | 0.6184 | 499,020,000 |
| SCAFFOLD | 20260615 | 0.5925 | 0.7041 | 0.4478 | 0.5952 | 0.7226 | 991,520,000 |
| SCAFFOLD | 20260616 | 0.5999 | 0.7123 | 0.4582 | 0.5896 | 0.7002 | 991,520,000 |
| SCAFFOLD | 20260617 | 0.6058 | 0.7064 | 0.4235 | 0.5918 | 0.7112 | 991,520,000 |

## Aggregate Results

| Method | Validation Macro-F1 mean | Validation Macro-F1 std | Validation user mean Macro-F1 mean | Validation worst 10% user Macro-F1 mean | Test Macro-F1 mean | Test Macro-F1 std | Test user mean Macro-F1 mean | Total communication mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAFFOLD | 0.5994 | 0.0067 | 0.7076 | 0.4432 | 0.5922 | 0.0028 | 0.7113 | 991,520,000 |
| FedAvg | 0.4060 | 0.0171 | 0.5598 | 0.3166 | 0.4280 | 0.0090 | 0.6324 | 499,020,000 |
| FedProx | 0.4039 | 0.0147 | 0.5572 | 0.3116 | 0.4249 | 0.0103 | 0.6294 | 499,020,000 |

Additional aggregate fairness snapshots:

| Method | Validation worst 10% user Macro-F1 std | Test user mean Macro-F1 std | Test worst 10% user Macro-F1 mean | Test worst 10% user Macro-F1 std |
|---|---:|---:|---:|---:|
| SCAFFOLD | 0.0178 | 0.0112 | 0.4031 | 0.0088 |
| FedAvg | 0.0043 | 0.0157 | 0.3381 | 0.0205 |
| FedProx | 0.0057 | 0.0125 | 0.3408 | 0.0249 |

## Interpretation

The sensitivity pass strengthens the earlier single-seed conclusion:
SCAFFOLD remains clearly ahead of FedAvg and FedProx across all three seeds.

Mean validation Macro-F1:

- SCAFFOLD: 0.5994
- FedAvg: 0.4060
- FedProx: 0.4039

Mean test Macro-F1:

- SCAFFOLD: 0.5922
- FedAvg: 0.4280
- FedProx: 0.4249

FedAvg and FedProx remain effectively tied. FedProx does not show a stable
improvement over FedAvg under this selected setting; the small per-seed
differences are within the observed seed variation.

SCAFFOLD has roughly twice the communication of FedAvg/FedProx at 50 rounds
because it exchanges control variates as well as model states. Even with that
caveat, the effect size is large enough that SCAFFOLD should be treated as the
current strongest client-drift baseline in the experiment matrix.

## Artifacts

- Runner: `kuhar_delivery/scripts/run_kuhar_seed_sensitivity.py`
- Raw per-seed summary: `outputs/kuhar_seed_sensitivity_v1/seed_results.csv`
- Aggregate summary: `outputs/kuhar_seed_sensitivity_v1/seed_summary.csv`

Recommended next step: either expand this pass to 5 seeds for paper-grade
confidence intervals, or proceed to the next algorithm and reserve 5-seed runs
for the final shortlist.
