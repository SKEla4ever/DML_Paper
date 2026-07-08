# Ditto

Ditto baseline for the frozen KU-HAR V1 minimum-support cohort.

Ditto trains a global FL model and a client-local personalized model for each
client. The global model is trained with FedAvg. Each personalized model is
trained locally with a proximal regularizer toward the current global model:

`cross_entropy(v_k) + lambda / 2 * ||v_k - w_global||^2`

Only global model updates are uploaded/downloaded. Personalized models stay on
clients and are used for known-client evaluation. Validation/test data are not
used for adaptation.

## Starting Point

- Global optimizer: Adam
- Personalized optimizer: Adam
- Learning rate: 0.001
- Global local epochs: 2
- Personalized local epochs: 2
- Batch size: 64
- Client fraction: 1.0
- Normalization: BatchNorm

## Example

```bash
python3 algorithms/ditto/train_kuhar_ditto.py \
  --cohort minimum_support \
  --rounds 50 \
  --eval-every 10 \
  --client-fraction 1.0 \
  --global-local-epochs 2 \
  --personal-local-epochs 2 \
  --batch-size 64 \
  --optimizer adam \
  --lr 0.001 \
  --ditto-lambda 0.1 \
  --norm batchnorm \
  --output-dir outputs/kuhar_ditto_lambda0p1_r50
```
