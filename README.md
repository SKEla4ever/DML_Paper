# DML Paper

Project repository for federated wearable HAR experiments.

## Current Frozen Data Identity

The current KU-HAR work uses the frozen V1 manifest in:

- `kuhar_delivery/data/processed/kuhar/`
- `kuhar_delivery/reports/kuhar/kuhar_recording_split_audit_v1.md`

Raw archives are intentionally not tracked by Git. Place KU-HAR V5 at:

`kuhar_delivery/data/raw/kuhar/2.Trimmed_interpolated_data.zip`

Expected SHA-256:

`9fe5d0052f2f1d6711afac42ee4badd968116afa8ba4b8ba591f4fdd771c2ec2`

## Algorithm Layout

Each implemented method lives under `algorithms/<method_name>/`.

Current folders:

- `algorithms/fedavg_softmax_sanity/`
- `algorithms/1d_cnn_fedavg/`

Generated outputs go under `outputs/` and are ignored by Git.

