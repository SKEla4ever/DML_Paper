# HHAR Manifest Audit V1

## Frozen identity

- Source: UCI HHAR, DOI `10.24432/C5689X`, license `CC BY 4.0`.
- Primary modality: smartphone three-axis accelerometer only.
- Client: physical-user/device pair; physical user is retained separately.
- Source archive SHA-256: `d4c0c53b195b523859bf71f5a349d164c7a604a321ff6b0972fbed6e03b46582`.
- Raw phone accelerometer rows: 13,062,475; users: 9; devices: 8; models: 4; observed user-device pairs: 71.

## Preprocessing protocol

`null` rows define label boundaries but are excluded from supervised windows. Within each non-null run, samples are ordered by sensor `Creation_Time`; gaps greater than 1 second create hard continuous-segment boundaries. Each segment is linearly resampled to 50 Hz and split into non-overlapping 3-second (150-sample) windows. No window crosses a label transition or sensor gap.

For each physical user/activity stratum, the majority device run count and median arrival intervals define canonical synchronized executions. Complete devices are matched by temporal ordinal; incomplete devices are matched by interval overlap. A raw run spanning multiple canonical executions is excluded as ambiguous. Every retained phone view of an execution is assigned to one split, preventing synchronized-motion leakage.

## Realized dataset

- Synchronized executions: 130; continuous segments: 2,041; included/excluded segments: 1,865/176.
- Phones represented per synchronized execution: 6–8; executions with a repeated same-device label run: 0.
- Raw runs excluded because their arrival interval ambiguously covered multiple canonical executions: 1.
- Supervised windows: 33,040; train/validation/test: 20,364/6,367/6,309 (61.63%/19.27%/19.10%).
- Retained user-device clients: 69; clients with train data: 69; clients meeting the optional >=3 supported test-class metric rule: 14.
- All six activities have support in every split: True.
- Weak-device condition identified by <25% of median device windows: s3mini_2.

## Interpretation boundary

HHAR V1 is a targeted device feature-shift benchmark. Its user-device pairs are repeated measurements from only nine physical users, so pair-level dispersion must not be presented as independent-user fairness evidence. The weak device remains in the primary full-device cohort and must be named when interpreting quantity skew or FedBN behavior.

## Integrity

- ZIP CRC: PASS; malformed rows: 0; non-finite rows: 0.
- Split unit is the synchronized execution; each execution appears in exactly one split and all included clients retain train data.
- Artifact SHA-256 values are recorded in `hhar_split_audit_v1.json`.

## Source

- https://archive.ics.uci.edu/dataset/344/heterogeneity%2Bactivity%2Brecognition
