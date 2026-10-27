# Stanford Dogs dataset audit

## Source

- Dataset: Stanford Dogs, 120 breeds.
- Original paper: Khosla et al., FGVC Workshop at CVPR 2011.
- Retrieval mirror: Hugging Face `amaye15/stanford-dogs`.
- Downloaded payload: 746,500,488 bytes across three parquet shards.

## Split audit

| Split | Images | Classes |
|---|---:|---:|
| train | 13,686 | 120 |
| val / calibration | 600 | 120 |
| test | 6,174 | 120 |

- The official test membership is unchanged.
- Exactly five retained train images per class form the clean validation and
  PMRC-calibration set.
- The remaining train images form the noisy classifier-training set.

## Duplicate-image audit

- Raw image bytes were hashed with SHA-256.
- Exactly 120 train-side images were removed because their hash occurred in
  official test or more than once in train.
- Four duplicate groups remain entirely inside official test. Three are
  same-class pairs and one is a cross-class pair. They are retained because
  the official test membership is preserved, and they do not create
  train/test leakage.
- No duplicate group connects train or validation to test after filtering.

## Noise audit

- Noise type: semantic neighbor.
- Requested rate: 40%.
- Empirical rate for each of the three noise seeds: 39.97%.
- Affected images per seed: 5,470 of 13,686 training images.

## Frozen artifacts

The protocol hash, fixed manifest hash, and seed-0 noise hash are stored in
`stanford_dogs_preregistered_artifacts.sha256`.
