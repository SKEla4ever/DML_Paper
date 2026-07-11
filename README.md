# DML Paper

Project repository for federated wearable HAR experiments.

## Frozen Data Identities

KU-HAR V1:

- `kuhar_delivery/data/processed/kuhar/`
- `kuhar_delivery/reports/kuhar/kuhar_recording_split_audit_v1.md`

HHAR V1:

- `hhar_delivery/data/processed/hhar/`
- `hhar_delivery/reports/hhar/hhar_manifest_audit_v1.md`
- `hhar_delivery/reports/hhar/hhar_diagnostic_v1.md`

Raw archives are intentionally not tracked by Git. Expected local paths:

- KU-HAR: `kuhar_delivery/data/raw/kuhar/2.Trimmed_interpolated_data.zip`
- HHAR: `hhar_delivery/data/raw/hhar/Activity recognition exp.zip`

Expected SHA-256 values:

- KU-HAR: `9fe5d0052f2f1d6711afac42ee4badd968116afa8ba4b8ba591f4fdd771c2ec2`
- HHAR: `d4c0c53b195b523859bf71f5a349d164c7a604a321ff6b0972fbed6e03b46582`

## Algorithm Layout

Each implemented method lives under `algorithms/<method_name>/`.

Current folders:

- `algorithms/fedavg_softmax_sanity/`
- `algorithms/1d_cnn_fedavg/`
- `algorithms/fedprox/`
- `algorithms/scaffold/`
- `algorithms/fedbn/`
- `algorithms/fedrep/`
- `algorithms/ditto/`

Generated outputs go under `outputs/` and are ignored by Git.
