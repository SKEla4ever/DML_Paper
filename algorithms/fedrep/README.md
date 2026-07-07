# FedRep

FedRep baseline for the frozen KU-HAR V1 minimum-support cohort.

FedRep learns a shared representation and client-local classifier heads. In
this 1D-CNN implementation, the representation is every layer before the final
`Linear` classifier, and the local head is the final `Linear` layer.

Each selected client:

1. Receives the current shared representation.
2. Updates only its local head for several local epochs.
3. Freezes the head and takes representation update steps.
4. Sends only the updated representation back to the server.

Evaluation is personalized for known clients: each subject is evaluated with
the shared representation plus that subject's local head. Validation/test data
are not used for adaptation.

## Starting Point

- Normalization: BatchNorm
- Optimizer: Adam
- Learning rate: 0.001
- Head epochs: 5
- Representation steps: 1
- Batch size: 64
- Client fraction: 1.0
- Rounds: 50

## Example

```bash
python3 algorithms/fedrep/train_kuhar_fedrep.py \
  --cohort minimum_support \
  --rounds 50 \
  --eval-every 10 \
  --client-fraction 1.0 \
  --head-epochs 5 \
  --representation-steps 1 \
  --batch-size 64 \
  --optimizer adam \
  --lr 0.001 \
  --norm batchnorm \
  --output-dir outputs/kuhar_fedrep_h5_rep1_lr0p001_r50
```
