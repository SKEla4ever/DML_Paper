# KU-HAR 1D CNN FedAvg Baseline V1

## Status

This is the first neural FedAvg checkpoint on the frozen KU-HAR V1 split.

Implementation:

`algorithms/1d_cnn_fedavg/train_kuhar_1d_cnn_fedavg.py`

Output directory:

`outputs/kuhar_1d_cnn_fedavg_minimum_support_v1`

Generated outputs are not tracked by Git.

## Model

Input:

- 3 accelerometer channels
- 300 samples per 3-second window

Architecture:

- Conv1d(3 -> 32), BatchNorm, ReLU, MaxPool
- Conv1d(32 -> 64), BatchNorm, ReLU, MaxPool
- Conv1d(64 -> 64), BatchNorm, ReLU
- AdaptiveAvgPool1d
- Dropout
- Linear classifier

## Run Settings

- Cohort: minimum-support
- Clients: 50
- Labels: 17 (`0`-`16`)
- Rounds: 20
- Client fraction: 1.0
- Local epochs: 1
- Batch size: 64
- Local optimizer: Adam
- Learning rate: 0.001
- Seed: `20260615`
- Raw-window cache: `outputs/cache/kuhar_v1_minimum_support_raw_accel_windows.npz`

## Communication Accounting

- Selected clients per round: 50
- Model-state bytes per client exchange: 99,804
- Total uplink: 99,804,000 bytes
- Total downlink: 99,804,000 bytes
- Total communication: 199,608,000 bytes

Communication counts the full PyTorch `state_dict` payload exchanged in both
uplink and downlink.

## Final Metrics

Round 20:

- Train accuracy: 0.3926
- Train Macro-F1: 0.2640
- Train per-user mean Macro-F1: 0.3165
- Validation accuracy: 0.4267
- Validation Macro-F1: 0.2770
- Validation per-user mean Macro-F1: 0.4461
- Test accuracy: 0.3825
- Test Macro-F1: 0.2521
- Test per-user mean Macro-F1: 0.4233
- Test worst 10% per-user Macro-F1: 0.1428

## Notes

This is a real neural FedAvg run, but it should still be treated as a first
checkpoint rather than a tuned final baseline. It currently underperforms the
softmax sanity baseline on test Macro-F1, likely because this first CNN setup
uses un-tuned local optimization and aggregates BatchNorm statistics through
plain FedAvg.

Next tuning targets:

- compare SGD and Adam systematically;
- try more rounds and lower evaluation frequency;
- test GroupNorm or local BatchNorm handling;
- tune learning rate and local epochs;
- add a centralized neural sanity baseline to separate model capacity from
federated optimization effects.
