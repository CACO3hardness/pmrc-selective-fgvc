# Prototype-source ablation

Lower AURC is better. Noisy prototypes use observed training labels.
Clean-oracle prototypes use unavailable clean training labels and are a
diagnostic upper bound, not a deployable method.

| Setting | n | No prototype | Noisy prototype | Clean oracle | ImageNet prototype | Shuffled prototype | Noisy wins vs no/ImageNet/shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Aircraft semantic-40 | 1 | 0.2337 | 0.2153 | 0.1830 | 0.2300 | 0.2341 | 1/1, 1/1, 1/1 |
| CUB semantic-40 | 3 | 0.2052 | 0.1769 | 0.1466 | 0.1812 | 0.2046 | 3/3, 2/3, 3/3 |
| Flowers102 semantic-37.5 | 1 | 0.1674 | 0.1347 | 0.0966 | 0.1292 | 0.1625 | 1/1, 0/1, 1/1 |

Paired bootstrap confidence intervals are retained in
`prototype_source_summary.json`.
