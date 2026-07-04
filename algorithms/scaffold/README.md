# SCAFFOLD

SCAFFOLD baseline for the frozen KU-HAR V1 minimum-support cohort.

SCAFFOLD uses the same 1D-CNN backbone, frozen manifest loader, and metrics as
`algorithms/1d_cnn_fedavg`, but local optimization uses control variates to
correct client drift:

`w <- w - lr * (grad - c_client + c_server)`

Communication accounting includes both model exchange and control-variate
exchange.

## Starting Point

Use the tuned FedAvg communication-efficient setting as the architectural
anchor:

- Normalization: BatchNorm
- Local epochs: 2
- Batch size: 64
- Client fraction: 1.0

SCAFFOLD uses corrected SGD-style local updates rather than Adam.

The first KU-HAR V1 sweep selected learning rate `1.0` by validation
Macro-F1. Smaller rates such as `0.01` are valid smoke-test values but
under-train this setup.

## Example

```bash
python3 algorithms/scaffold/train_kuhar_scaffold.py \
  --cohort minimum_support \
  --rounds 50 \
  --eval-every 10 \
  --client-fraction 1.0 \
  --local-epochs 2 \
  --batch-size 64 \
  --lr 1.0 \
  --norm batchnorm \
  --output-dir outputs/kuhar_scaffold_lr1p0_r50
```

For additional sweeps:

```bash
python3 algorithms/scaffold/run_lr_grid.py \
  --rounds 20 \
  --eval-every 5 \
  --local-epochs 2 \
  --batch-size 64 \
  --lr-values 0.5 0.8 1.0
```
