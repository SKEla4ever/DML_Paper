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

The diagnostic uses a reduced, explicitly labeled setup-check budget. Final
FedAvg selection still requires split-seed sensitivity, validation-only tuning,
and a model-seed pass.

Raw archives and generated training caches remain local and are not committed.
