# Decision-cost analysis

Error cost is normalized to one. Rejection cost is the cost of abstaining
relative to one classification error. Thresholds are selected on
calibration data and evaluated on fixed test data.

| Rejection cost | Evaluations | Mean MSP cost | Mean PMRC cost | PMRC lower | MSP coverage | PMRC coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 40 | 0.0842 | 0.0802 | 33/40 | 30.3% | 34.2% |
| 0.25 | 40 | 0.1786 | 0.1687 | 36/40 | 47.4% | 50.0% |
| 0.50 | 40 | 0.2831 | 0.2692 | 36/40 | 73.3% | 72.8% |
| 1.00 | 40 | 0.3251 | 0.3249 | 24/40 | 99.7% | 99.7% |
