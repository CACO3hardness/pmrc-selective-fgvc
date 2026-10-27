# Selective-baseline comparison

PMRC lower indicates that PMRC had lower test AURC than the baseline
on the same model and test samples.

| Evaluation set | Method | Available runs | Mean AURC | PMRC lower |
|---|---|---:|---:|---:|
| Controlled 28-model matrix | Temperature | 28 | 0.1547 | 28/28 |
| Controlled 28-model matrix | MaxLogit-pNorm | 28 | 0.1517 | 27/28 |
| Controlled 28-model matrix | KNN agreement | 28 | 0.1553 | 27/28 |
| Controlled 28-model matrix | Trust Score | 28 | 0.1734 | 28/28 |
| WebFG real web noise | Temperature | 12 | 0.0889 | 9/12 |
| WebFG real web noise | MaxLogit-pNorm | 12 | 0.0877 | 5/12 |
| WebFG real web noise | KNN agreement | 12 | 0.0933 | 9/12 |
| WebFG real web noise | Trust Score | 12 | 0.0859 | 4/12 |
