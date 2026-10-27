# Stanford Dogs external-domain validation protocol

## Status

Frozen before any Stanford Dogs classifier is trained or evaluated. Do not
change the class set, split rule, noise rate, architecture, seeds, aggregation,
or gate threshold after observing model or test outcomes.

## Why this dataset

WebiNat-5089 remains the preferred real-web external domain, but its full
archive is disproportionately large for the remaining validation question.
Stanford Dogs is an independent fine-grained dataset that was not used to
design PMRC or its gate. It tests cross-domain transport under the same frozen
post-hoc procedure.

This experiment validates an independent external fine-grained domain under
synthetic semantic-neighbor noise. It does not replace a real-web external
domain and must not be described as such.

## Source and fixed partitions

- Dataset: Stanford Dogs, 120 dog breeds.
- Upstream source: Khosla et al., FGVC Workshop at CVPR 2011.
- Public mirror used for retrieval: Hugging Face
  `amaye15/stanford-dogs`.
- Official split sizes: 14,406 train and 6,174 test images.
- Preserve the official train/test membership.
- Before splitting, compute SHA-256 on the raw image bytes. Remove every train
  image whose hash also occurs in the official test set or occurs more than
  once within train. Keep the official test set intact and record all removed
  train images in the manifest metadata. This prevents exact train/test
  leakage identified during preparation.
- The final manifest contains 13,686 training, 600 validation/calibration, and
  6,174 test images. Exactly 120 duplicate train-side images were removed.
  Four duplicate groups remain wholly inside the official test set and are
  left intact to preserve that split; no duplicate group crosses train or
  validation with test.
- For each class, order the official train images by SHA-256 of their relative
  source identifier and reserve exactly five images for the clean validation
  and PMRC-calibration partition. The remaining train images form the noisy
  classifier-training partition.
- Preserve every official test image in the fixed test partition. Official
  test images are not used for checkpoint selection or PMRC fitting.
- The exact image, class, and split assignment is shared by all classifier
  seeds. Only model initialization and semantic-noising seed change.

## Noise construction

- Noise kind: semantic neighbor.
- Requested corruption rate: 40%.
- Neighbor candidates: top five pretrained class centroids.
- Apply noise only to the classifier-training partition after the clean
  validation partition has been removed.
- Report the empirical observed noise rate and keep all classes.

## Classifier training

- Backbone: ImageNet-pretrained ResNet-50.
- Input: 224 x 224.
- Optimizer procedure: the same AdamW, cosine schedule, batch size, and
  early-learning regularization settings used for the other ResNet-50
  experiments.
- Epochs: 20.
- Seeds: 0, 1, and 2.
- Select each checkpoint by clean validation accuracy.

## PMRC and gate

- Use the frozen PMRC implementation and ten-descriptor feature set.
- Build prototypes from the observed noisy training labels.
- Fit correctness calibration and its L2 strength on the fixed 600-image clean
  validation/calibration split only.
- Compute the applicability diagnostic exactly as in the manuscript, using
  five equal-frequency MSP strata and within-stratum prototype-margin AUROC.
- Apply the unchanged gate threshold of 0.5.
- Evaluate MSP, gated PMRC, direct PMRC, Temperature, MaxLogit-pNorm, KNN, and
  Trust Score on the fixed clean test partition.

## Primary outcomes

1. Seed-level and mean test AURC for MSP, direct PMRC, and gated PMRC.
2. Paired test-image bootstrap interval for PMRC minus MSP in each seed.
3. Gate diagnostic, gate decision, and whether its direction matches the
   direct PMRC effect.
4. Selective risk at 90% coverage.

## Pre-specified interpretation

- A successful external-domain result requires no seed-level gated regression
  and the correct gate direction in all three seeds.
- A positive mean with any gated regression is partial success.
- A sign mismatch is a negative result about gate transport.
- Report all seeds. Do not remove, replace, or rerun a seed after its test
  result is known.
- PMRC remains a closed-set method. No open-set claim follows from this
  experiment.

## Manuscript placement

If the result is positive, add one concise external-domain paragraph and a
small supplementary seed-level table. If it is mixed or negative, report it as
a scope-limiting result and retain the current conditional claims.
