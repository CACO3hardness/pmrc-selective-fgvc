# Calibration-set size stability

Each row replays random subsets of the clean validation set, refits the full-feature PMRC calibrator, and evaluates the unchanged test set. Values show mean AURC and seed-to-seed or subset-to-subset standard deviation where repeated.

| Dataset | Run | 25% samples | 50% samples | 75% samples | 100% samples |
|---|---|---:|---:|---:|---:|
| CUB semantic40 | seed0 | 0.1721 ± 0.0045 (n=150) | 0.1678 ± 0.0021 (n=300) | 0.1651 ± 0.0009 (n=450) | 0.1643 (n=600) |
| CUB semantic40 | seed1 | 0.2062 ± 0.0073 (n=150) | 0.1973 ± 0.0027 (n=300) | 0.1963 ± 0.0014 (n=450) | 0.1952 (n=600) |
| Aircraft semantic40 | seed0 | 0.2226 ± 0.0048 (n=175) | 0.2165 ± 0.0016 (n=350) | 0.2147 ± 0.0008 (n=525) | 0.2141 (n=700) |
| Flowers102 semantic37.5 | seed0 | 0.1620 ± 0.0156 (n=51) | 0.1406 ± 0.0071 (n=102) | 0.1342 ± 0.0029 (n=153) | 0.1337 (n=204) |
| CIFAR-100N human40.2 | seed0 | 0.1551 ± 0.0032 (n=250) | 0.1516 ± 0.0013 (n=500) | 0.1505 ± 0.0010 (n=750) | 0.1502 (n=1000) |

## Interpretation

- PMRC remains below MSP at every tested calibration size, including the smallest 51-image Flowers102 subset.
- Stability increases with calibration-set size.
- Flowers102 is the most sensitive setting because its 25% subset contains only 51 images.
- The main paper should report the full-validation results and place this table in Supplementary Information.

