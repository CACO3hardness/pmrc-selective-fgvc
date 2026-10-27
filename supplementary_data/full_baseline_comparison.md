# Full baseline comparison

All values are unweighted means across the 28 completed model evaluations. Lower AURC is better. Each baseline uses the same test samples as PMRC within a model.

| Method | Mean AURC | PMRC lower in |
|---|---:|---:|
| PMRC | **0.1358** | — |
| Temperature Scaling | 0.1547 | 28/28 |
| MaxLogit-pNorm | 0.1517 | 27/28 |
| KNN label agreement | 0.1553 | 27/28 |
| Trust Score | 0.1734 | 28/28 |

## Exceptions

- MaxLogit-pNorm has slightly lower AURC on the clean CUB control, 0.0435 versus PMRC 0.0447.
- KNN agreement has slightly lower AURC on CUB symmetric-40 with cross-entropy, 0.1681 versus PMRC 0.1689.

Both exceptions are reported in the manuscript. Neither changes the central noisy-label conclusion.

## WebFG real web noise

The twelve WebFG runs show a different ranking. PMRC mean AURC was 0.0873,
compared with 0.0889 for temperature scaling, 0.0877 for MaxLogit-pNorm,
0.0933 for KNN agreement, and 0.0859 for Trust Score. PMRC was lower than the
four baselines in 9/12, 5/12, 9/12, and 4/12 comparisons, respectively. The paper
therefore reports WebFG as conditional external evidence, not as uniform
state-of-the-art superiority.

See `selective_baseline_comparison_final.{json,md}` for machine-readable counts
and exceptions.
