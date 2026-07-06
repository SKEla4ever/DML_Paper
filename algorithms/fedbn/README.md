# FedBN

FedBN baseline for the frozen KU-HAR V1 minimum-support cohort.

FedBN uses the same 1D-CNN backbone, frozen manifest loader, and metrics as
`algorithms/1d_cnn_fedavg`, but keeps BatchNorm state local to each client.
Only non-BatchNorm model state is uploaded, averaged, and downloaded.

Evaluation is client-specific: each subject is evaluated with the shared global
Conv/Linear weights plus that subject's local BatchNorm state.

## Selected Starting Point

Use the tuned FedAvg communication-efficient setting as the base:

- Normalization: BatchNorm
- Optimizer: Adam
- Learning rate: 0.001
- Local epochs: 2
- Batch size: 64
- Client fraction: 1.0
- Rounds: 50

## Example

```bash
python3 algorithms/fedbn/train_kuhar_fedbn.py \
  --cohort minimum_support \
  --rounds 50 \
  --eval-every 10 \
  --client-fraction 1.0 \
  --local-epochs 2 \
  --batch-size 64 \
  --optimizer adam \
  --lr 0.001 \
  --output-dir outputs/kuhar_fedbn_lr0p001_e2_r50
```
