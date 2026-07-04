# FedProx

FedProx baseline for the frozen KU-HAR V1 minimum-support cohort.

FedProx uses the same 1D-CNN backbone, frozen manifest loader, metrics, and
communication accounting as `algorithms/1d_cnn_fedavg`. The only algorithmic
change is the local proximal objective:

`loss = cross_entropy + (mu / 2) * ||w_local - w_global||^2`

First tuning report:

`kuhar_delivery/reports/kuhar/kuhar_fedprox_v1.md`

## Current Starting Point

Use the communication-efficient FedAvg setting as the starting point:

- Normalization: BatchNorm
- Optimizer: Adam
- Learning rate: 0.001
- Local epochs: 2
- Batch size: 64

## Example

```bash
python3 algorithms/fedprox/train_kuhar_fedprox.py \
  --cohort minimum_support \
  --rounds 50 \
  --eval-every 10 \
  --client-fraction 1.0 \
  --local-epochs 2 \
  --batch-size 64 \
  --optimizer adam \
  --lr 0.001 \
  --norm batchnorm \
  --mu 0.01 \
  --output-dir outputs/kuhar_fedprox_mu0p01_r50
```
