# Stanford Dogs external-domain validation

The official train/test split is fixed. Five official train images per class form the clean validation/calibration set, and the remaining training labels receive 40% semantic-neighbor noise. The official test set is untouched.

| Seed | MSP AURC | PMRC AURC | Relative gain | 95% CI | Gate AUROC | Gate |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.0861 | 0.0614 | +28.63% | [-0.0287, -0.0208] | 0.8526 | PMRC |
| 1 | 0.0799 | 0.0588 | +26.46% | [-0.0244, -0.0178] | 0.7854 | PMRC |
| 2 | 0.0897 | 0.0627 | +30.11% | [-0.0317, -0.0230] | 0.8162 | PMRC |

Mean relative gain: +28.40%. Positive seeds: 3/3. Gate direction matches: 3/3. PMRC beats all four baselines in 3/3.

| Baseline | Mean AURC |
|---|---:|
| Temperature | 0.0801 |
| MaxLogit-pNorm | 0.0809 |
| KNN agreement | 0.0743 |
| Trust Score | 0.1237 |
| PMRC | 0.0610 |

Stanford Dogs is derived from ImageNet and the backbone is ImageNet-pretrained. This experiment therefore measures downstream dataset and split transport under synthetic semantic noise, not transport to a pretraining-independent real-web domain.
