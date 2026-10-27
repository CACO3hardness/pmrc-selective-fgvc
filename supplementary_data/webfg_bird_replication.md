# WebFG bird replication

All completed seeds are reported. The gate uses validation within-MSP
prototype AUROC, while the final column reports the held-out test value.

| Seed | Acc | MSP AURC | PMRC AURC | Relative change | MSP risk@90 | PMRC risk@90 | Validation AUC | Test AUC | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 76.10% | 0.07336 | 0.07424 | -1.20% | 0.18639 | 0.18490 | 0.3630 | 0.4547 | msp |
| 1 | 75.87% | 0.07839 | 0.07891 | -0.67% | 0.18895 | 0.18874 | 0.4231 | 0.4766 | msp |
| 2 | 75.11% | 0.08388 | 0.08219 | +2.01% | 0.20154 | 0.19599 | 0.4879 | 0.4622 | msp |
| 3 | 76.26% | 0.07615 | 0.07840 | -2.96% | 0.18575 | 0.18298 | 0.4328 | 0.4394 | msp |
| 4 | 75.93% | 0.07438 | 0.07441 | -0.04% | 0.18746 | 0.18426 | 0.4441 | 0.4343 | msp |
| 5 | 75.55% | 0.07767 | 0.07781 | -0.19% | 0.19023 | 0.18682 | 0.4448 | 0.4619 | msp |

## Group summaries

- `original_seed0_2`: n=3, mean seed-wise relative change +0.05% ± 1.72% (-1.20% to +2.01%), 1/3 seeds improved, gate selected PMRC in 0/3, mean 90% risk reduction +0.24 pp.
- `replication_seed3_5`: n=3, mean seed-wise relative change -1.06% ± 1.64% (-2.96% to -0.04%), 0/3 seeds improved, gate selected PMRC in 0/3, mean 90% risk reduction +0.31 pp.
- `all_available`: n=6, mean seed-wise relative change -0.51% ± 1.62% (-2.96% to +2.01%), 1/6 seeds improved, gate selected PMRC in 0/6, mean 90% risk reduction +0.28 pp.
