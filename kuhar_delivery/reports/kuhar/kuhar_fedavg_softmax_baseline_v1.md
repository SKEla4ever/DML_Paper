# KU-HAR FedAvg Softmax Baseline V1

## Status

This is the first end-to-end training baseline on the frozen KU-HAR V1 split.
It uses `kuhar_delivery/scripts/train_kuhar_fedavg_numpy.py`, which implements:

- frozen manifest loading;
- raw accelerometer window extraction from `2.Trimmed_interpolated_data.zip`;
- statistical window features;
- NumPy softmax classifier;
- FedAvg over subject-level clients;
- full-precision uplink and downlink communication accounting;
- global and per-user sanity metrics.

This is a pipeline sanity baseline, not the final neural HAR model.

## Source Check

- Raw archive: `kuhar_delivery/data/raw/kuhar/2.Trimmed_interpolated_data.zip`
- Size: 161,492,605 bytes
- SHA-256: `9fe5d0052f2f1d6711afac42ee4badd968116afa8ba4b8ba591f4fdd771c2ec2`
- Checksum status: matched frozen KU-HAR V1 source.

## Manifest Sanity

Minimum-support cohort:

- Clients: 50
- Labels: 17 (`0`-`16`)
- Windows: 13,627
- Train/validation/test windows: 8,177 / 2,632 / 2,818

Full sparse cohort:

- Clients: 88 training clients
- Evaluable clients: 54
- Labels: 18 (`0`-`17`)
- Windows: 19,641
- Train/validation/test windows: 12,351 / 3,578 / 3,712

## Real Baseline Runs

Shared run settings:

- Rounds: 20
- Client fraction: 1.0
- Local epochs: 1
- Batch size: 32
- Learning rate: 0.1
- Seed: `20260615`
- Feature count: 32

### Minimum-Support Main Cohort

Output directory: `outputs/kuhar_fedavg_minimum_support_v1`

- Selected clients per round: 50
- Parameters: 561
- Bytes per full-precision model: 2,244
- Total communication: 4,488,000 bytes
- Test accuracy: 0.5000
- Test Macro-F1: 0.3864
- Test per-user mean Macro-F1: 0.5386
- Test worst 10% per-user Macro-F1: 0.2431

Round 0 to round 20:

- Test accuracy: 0.0277 -> 0.5000
- Test Macro-F1: 0.0306 -> 0.3864
- Test per-user mean Macro-F1: 0.0387 -> 0.5386

### Full Sparse Sensitivity Cohort

Output directory: `outputs/kuhar_fedavg_full_sparse_v1`

- Selected clients per round: 88
- Parameters: 594
- Bytes per full-precision model: 2,376
- Total communication: 8,363,520 bytes
- Test accuracy: 0.0892
- Test Macro-F1: 0.0670
- Test per-user mean Macro-F1: 0.0617
- Test worst 10% per-user Macro-F1: 0.0000

Round 0 to round 20:

- Test accuracy: 0.0321 -> 0.0892
- Test Macro-F1: 0.0210 -> 0.0670
- Test per-user mean Macro-F1: 0.0222 -> 0.0617

## Smoke Test

Output directory: `outputs/kuhar_fedavg_synthetic_smoke`

The synthetic-smoke run uses deterministic synthetic features and exists only to
verify the FedAvg loop, metrics, and output writing. It must not be reported as
an experimental result.

## Interpretation

The minimum-support run confirms that the frozen V1 loader, client grouping,
FedAvg update loop, communication accounting, and user-level metrics are
working on real accelerometer data.

The full sparse result is intentionally difficult and remains a stress
sensitivity setting: a simple softmax model over statistical features performs
poorly under the 18-class sparse protocol. This is useful as a sanity warning,
not as a final algorithm comparison.

Next implementation step: replace the softmax sanity model with the first
neural HAR baseline while preserving the same manifest loader, communication
accounting, and metric schema.
