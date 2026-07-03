# KU-HAR Training Pipeline V1

## Purpose

This is the first runnable training-pipeline checkpoint for the frozen KU-HAR
V1 manifest. It covers:

- data loading from `kuhar_window_split_manifest.csv.gz`;
- cohort filtering for `minimum_support` and `full_sparse`;
- baseline FedAvg training with a NumPy softmax classifier;
- full-precision communication accounting for uplink and downlink model
  exchange;
- sanity metrics: loss, accuracy, Macro-F1, per-user Macro-F1, worst 10%
  per-user Macro-F1, and user-level standard deviation.

The softmax model is a sanity baseline, not the final HAR architecture. The
later neural baseline should keep the same frozen split identity and reuse the
same metric and communication-accounting definitions.

## Raw Data Requirement

The script expects the frozen KU-HAR V5 archive at:

`kuhar_delivery/data/raw/kuhar/2.Trimmed_interpolated_data.zip`

Expected SHA-256:

`9fe5d0052f2f1d6711afac42ee4badd968116afa8ba4b8ba591f4fdd771c2ec2`

The current manifest files are enough for split validation, but real training
requires the raw archive because the window manifest stores offsets and labels,
not accelerometer values.

Current local status: the archive is present and its SHA-256 matches the frozen
V1 source.

## Commands

Manifest sanity check:

```bash
python3 kuhar_delivery/scripts/train_kuhar_fedavg_numpy.py \
  --dry-run \
  --output-dir outputs/kuhar_fedavg_manifest_sanity
```

End-to-end smoke test using deterministic synthetic features:

```bash
python3 kuhar_delivery/scripts/train_kuhar_fedavg_numpy.py \
  --synthetic-smoke \
  --rounds 3 \
  --output-dir outputs/kuhar_fedavg_synthetic_smoke
```

Real minimum-support baseline once the raw archive is present:

```bash
python3 kuhar_delivery/scripts/train_kuhar_fedavg_numpy.py \
  --cohort minimum_support \
  --rounds 20 \
  --client-fraction 1.0 \
  --local-epochs 1 \
  --batch-size 32 \
  --lr 0.1 \
  --output-dir outputs/kuhar_fedavg_minimum_support_v1
```

Full sparse sensitivity baseline:

```bash
python3 kuhar_delivery/scripts/train_kuhar_fedavg_numpy.py \
  --cohort full_sparse \
  --rounds 20 \
  --client-fraction 1.0 \
  --local-epochs 1 \
  --batch-size 32 \
  --lr 0.1 \
  --output-dir outputs/kuhar_fedavg_full_sparse_v1
```

## Outputs

Each run writes:

- `manifest_sanity.json`
- `run_config.json`
- `feature_standardization.json`
- `metrics_history.json`
- `round_metrics.csv`
- `final_metrics.json`
- `final_model.json`

Runs with `synthetic_smoke: true` are code smoke tests only and must not be
reported as experimental results.
