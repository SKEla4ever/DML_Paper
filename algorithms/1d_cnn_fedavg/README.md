# 1D CNN FedAvg

First neural FedAvg baseline for KU-HAR frozen V1.

The model consumes raw 3-axis accelerometer windows with shape:

`channels=3, samples=300`

The training script keeps the same frozen manifest identity, subject-level
client grouping, communication accounting, and per-user metric definitions as
the softmax sanity baseline.

First completed checkpoint:

`kuhar_delivery/reports/kuhar/kuhar_1d_cnn_fedavg_baseline_v1.md`

## Tuning Grid

```bash
python3 algorithms/1d_cnn_fedavg/run_tuning_grid.py \
  --rounds 20 \
  --eval-every 5 \
  --output-root outputs/kuhar_1d_cnn_fedavg_tuning_v1
```

The grid compares `BatchNorm` and `GroupNorm`, `Adam` and `SGD`, and a small
learning-rate set. Selection should use validation metrics only.

## Dependency

This method requires PyTorch.

## HHAR V1

The HHAR entry point reuses the same model, optimization, aggregation, and
communication accounting while loading the frozen synchronized-execution HHAR
manifest. HHAR windows have shape `channels=3, samples=150`; clients are
physical-user/device pairs.

```bash
python3 algorithms/1d_cnn_fedavg/train_hhar_1d_cnn_fedavg.py \
  --rounds 20 \
  --local-epochs 1 \
  --optimizer adam \
  --lr 0.001 \
  --output-dir outputs/hhar_1d_cnn_fedavg_v1
```

## Example

```bash
python3 algorithms/1d_cnn_fedavg/train_kuhar_1d_cnn_fedavg.py \
  --cohort minimum_support \
  --rounds 20 \
  --client-fraction 1.0 \
  --local-epochs 1 \
  --batch-size 32 \
  --lr 0.001 \
  --output-dir outputs/kuhar_1d_cnn_fedavg_minimum_support_v1
```
