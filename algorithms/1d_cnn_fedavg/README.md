# 1D CNN FedAvg

Neural 1D-CNN FedAvg baseline for the frozen KU-HAR and HHAR protocols.

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
.venv/bin/python algorithms/1d_cnn_fedavg/train_hhar_1d_cnn_fedavg.py \
  --manifest-dir \
    outputs/hhar_split_seed_sensitivity_v1/split_seed20260615/manifest \
  --rounds 50 \
  --local-epochs 1 \
  --optimizer sgd \
  --momentum 0.9 \
  --lr 0.01 \
  --norm batchnorm \
  --evaluation-splits train validation \
  --output-dir outputs/hhar_1d_cnn_fedavg_v1
```

The frozen primary HHAR setting was selected using aggregate validation
Macro-F1 over split seeds `20260615`, `20260616`, and `20260617`. Its report is
`hhar_delivery/reports/hhar/hhar_fedavg_lr_schedule_v1.md`. Test evaluation was
not run during tuning.

The follow-up model-seed study evaluates the frozen setting over a full
`3 split seeds x 5 model seeds` matrix. Its balanced validation Macro-F1 is
`0.4645`; model-seed marginal SD/range are `0.0398 / 0.0969`, while split-seed
marginal SD/range are `0.0075 / 0.0143`. The registered stability rule passes,
but the model-seed spread remains material and must be retained in subsequent
HHAR comparisons. All 15 checkpoints were frozen before a separately
pre-registered evaluation-only locked test.

Model-seed report:
`hhar_delivery/reports/hhar/hhar_fedavg_model_seed_expansion_v1.md`

The locked test is complete for all 15 checkpoints, with no retraining,
test-driven tuning, checkpoint selection, or seed selection. Balanced test
Macro-F1 is `0.6347`, model-seed marginal SD is `0.0180`, and split-seed
marginal SD is `0.0182`. This is the closed HHAR FedAvg test reference for
matching-budget method comparisons.

Locked-test report:
`hhar_delivery/reports/hhar/hhar_fedavg_locked_test_v1.md`

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
