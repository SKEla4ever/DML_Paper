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
split-seed sensitivity and validation-only FedAvg tuning are complete. A
separate model-seed pass remains; test metrics stay locked until that pass is
complete.

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
compared as if they were equal-budget runs. Hyperparameters are frozen; a
separate model-seed sensitivity pass is required before locked test evaluation.

Final selection report:
`hhar_delivery/reports/hhar/hhar_fedavg_lr_schedule_v1.md`

Raw archives and generated training caches remain local and are not committed.
