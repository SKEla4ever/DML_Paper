# HHAR Delivery

Frozen HHAR V1 pipeline for the targeted device feature-shift experiments.

## Protocol

- Source: UCI HHAR activity-recognition archive, DOI `10.24432/C5689X`.
- Modality: phone three-axis accelerometer.
- Client: physical-user/device pair; physical-user identity remains explicit.
- Split unit: synchronized physical-user activity execution across all phones.
- Synchronization: majority device run count plus median arrival-time canonical
  executions; one cross-execution ambiguous raw run is excluded.
- Windowing: 50 Hz resampling, 3-second non-overlapping windows.
- Gap policy: no interpolation across a sensor-time gap greater than 1 second.

## Regenerate Manifest

```bash
python3 hhar_delivery/scripts/generate_hhar_manifest.py \
  --archive "hhar_delivery/data/raw/hhar/Activity recognition exp.zip" \
  --output-dir hhar_delivery/data/processed/hhar \
  --report hhar_delivery/reports/hhar/hhar_manifest_audit_v1.md
```

## FedAvg Dry Run

```bash
python3 algorithms/1d_cnn_fedavg/train_hhar_1d_cnn_fedavg.py --dry-run
```

## Setup Diagnostic

```bash
python3 hhar_delivery/scripts/run_hhar_diagnostic_v1.py
```

The diagnostic uses a reduced, explicitly labeled setup-check budget. The
split-seed sensitivity, validation-only FedAvg tuning, and five-model-seed
characterization are complete. The separately pre-registered, evaluation-only
locked test is also complete; it is closed to post-test FedAvg tuning and seed
selection.

## Split-Seed Sensitivity

```bash
python3 hhar_delivery/scripts/run_hhar_split_seed_sensitivity_v1.py
```

V1 fixes split seeds `20260615`, `20260616`, and `20260617`, while holding the
model/optimizer seed at `20260615`. This pass is complete; its split identities
must be reused for HHAR hyperparameter selection.

Results: `hhar_delivery/reports/hhar/hhar_split_seed_sensitivity_v1.md`

## FedAvg Validation-Only Tuning

```bash
.venv/bin/python hhar_delivery/scripts/run_hhar_fedavg_3split_tuning_v1.py
.venv/bin/python hhar_delivery/scripts/run_hhar_fedavg_50round_refinement_v1.py
.venv/bin/python hhar_delivery/scripts/run_hhar_fedavg_lr_schedule_v1.py
```

All tuning runs evaluate only `train` and `validation`; the scripts reject any
run containing a test performance metric. The frozen references are:

- Primary high-budget: BatchNorm, SGD momentum `0.9`, constant learning rate
  `0.01`, 1 local epoch, 50 rounds, full participation.
- Communication-efficient supplementary: BatchNorm, SGD momentum `0.9`,
  constant learning rate `0.03`, 1 local epoch, 20 rounds, full participation.

The two references have different communication budgets and must not be
compared as if they were equal-budget runs. Hyperparameters are frozen.

Final selection report:
`hhar_delivery/reports/hhar/hhar_fedavg_lr_schedule_v1.md`

## FedAvg Model-Seed Sensitivity

```bash
.venv/bin/python \
  hhar_delivery/scripts/run_hhar_fedavg_model_seed_sensitivity_v1.py
.venv/bin/python \
  hhar_delivery/scripts/run_hhar_fedavg_model_seed_expansion_v1.py
```

The initial `3 splits x 3 model seeds` pass narrowly exceeded its registered
model-seed SD threshold, which triggered the pre-registered expansion to five
model seeds. The final `3 x 5` validation-only matrix has balanced Macro-F1
`0.4645`, model-seed marginal SD `0.0398`, and range `0.0969`; it passes the
registered `SD <= 0.05` and `range <= 0.10` rule. The range remains material,
so subsequent HHAR headline comparisons must retain the five-seed distribution
and report model-seed and split-seed variability separately.

All 15 final checkpoints are frozen and hash-recorded. They were subsequently
evaluated once under the separately pre-registered locked-test protocol without
retraining or seed selection.

Final sensitivity report:
`hhar_delivery/reports/hhar/hhar_fedavg_model_seed_expansion_v1.md`

## FedAvg Locked Test

The first and final locked test evaluated the complete `3 split seeds x 5 model
seeds` checkpoint matrix. The balanced test Macro-F1 is `0.6347` and Accuracy
is `0.6282`. Model-seed marginal SD/range are `0.0180 / 0.0465`; split-seed
marginal SD/range are `0.0182 / 0.0362`. The four confirmatory model-seed
marginals are `0.6376 +/- 0.0193`.

The balanced test-minus-validation Macro-F1 gap is `0.1702`, consistent with
the diagnostic finding that the frozen test executions are easier than the
validation executions. This test result is now closed: it must be used as the
HHAR FedAvg reference and must not drive further FedAvg tuning, checkpoint
selection, or seed selection.

Pre-registration:
`hhar_delivery/reports/hhar/hhar_fedavg_locked_test_v1_preregistration.json`

Final report and machine-readable summary:
`hhar_delivery/reports/hhar/hhar_fedavg_locked_test_v1.md` and
`hhar_delivery/reports/hhar/hhar_fedavg_locked_test_v1_summary.json`

Raw archives and generated training caches remain local and are not committed.
