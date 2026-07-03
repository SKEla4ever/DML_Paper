# 1D CNN FedAvg

First neural FedAvg baseline for KU-HAR frozen V1.

The model consumes raw 3-axis accelerometer windows with shape:

`channels=3, samples=300`

The training script keeps the same frozen manifest identity, subject-level
client grouping, communication accounting, and per-user metric definitions as
the softmax sanity baseline.

First completed checkpoint:

`kuhar_delivery/reports/kuhar/kuhar_1d_cnn_fedavg_baseline_v1.md`

## Dependency

This method requires PyTorch.

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
