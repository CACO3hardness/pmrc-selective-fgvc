# Expanded validity evidence

All analyses reuse frozen classifiers. No test labels are used to fit
calibrators, tune gates, or select thresholds.

## Calibration-label and class-coverage robustness

| Perturbation | Rate | Evaluations | Mean direct PMRC gain | Mean gated gain | Non-regression rate |
|---|---:|---:|---:|---:|---:|
| Clean calibration | 0% | 40 | +9.6% | +9.7% | 100.0% |
| Symmetric calibration-label noise | 5% | 200 | +9.1% | +9.2% | 99.0% |
| Symmetric calibration-label noise | 10% | 200 | +8.1% | +8.3% | 97.5% |
| Symmetric calibration-label noise | 20% | 200 | +6.3% | +6.7% | 88.0% |
| Missing calibration classes | 10% | 200 | +9.4% | +9.5% | 99.5% |
| Missing calibration classes | 25% | 200 | +9.0% | +9.1% | 97.0% |
| Missing calibration classes | 50% | 200 | +7.4% | +7.6% | 89.0% |

The clean parameter estimate is held fixed within each run. Only
the calibration labels or calibration-class coverage are perturbed.

## Calibration and gate transfer to unseen classes

| Held-out fraction | Evaluations | Mean direct PMRC gain | Mean gated gain | Direct non-regression | Gated non-regression |
|---:|---:|---:|---:|---:|---:|
| 20% | 200 | +8.1% | +8.5% | 81.0% | 90.5% |
| 40% | 200 | +7.9% | +8.4% | 81.0% | 91.0% |

For each split, calibration uses only the non-held-out classes and
evaluation uses only held-out classes. Repeats use five independent
class partitions.

## Split LTT risk-control baseline

Clopper-Pearson upper bounds use a 10% family-wise error budget
after a union correction over all candidate thresholds.

| Target risk | Score | Runs accepted | Mean test coverage | Mean test risk |
|---:|---|---:|---:|---:|
| 5% | MSP | 0/42 | 0.0% | not available |
| 5% | PMRC | 1/42 | 0.6% | 1.8% |
| 10% | MSP | 15/42 | 15.4% | 4.5% |
| 10% | PMRC | 22/42 | 22.2% | 4.3% |
| 15% | MSP | 27/42 | 31.8% | 7.3% |
| 15% | PMRC | 37/42 | 43.4% | 7.8% |

At the 5% target, the family-wise finite-sample bound accepts no
samples in most runs. The comparison is nonetheless useful because
it quantifies the price of a formal risk guarantee relative to the
empirical calibration used in the main experiments.

## Empirical and finite-sample risk violations

| Target risk | Empirical MSP violations | Empirical PMRC violations | LTT MSP violations | LTT PMRC violations |
|---:|---:|---:|---:|---:|
| 5% | 24/42 | 26/42 | 0 | 0 |
| 10% | 21/42 | 25/42 | 0 | 0 |
| 15% | 19/42 | 27/42 | 0 | 0 |

Empirical violations are counted over all runs, including runs that accept no samples. LTT violations are counted among runs that accept at least one sample.

Paired McNemar comparisons of empirical violations:

| Target risk | PMRC-only | MSP-only | Exact p |
|---:|---:|---:|---:|
| 5% | 9 | 7 | 0.804 |
| 10% | 9 | 5 | 0.424 |
| 15% | 10 | 2 | 0.039 |

## Gate-threshold sensitivity

| Scope | Threshold | Mean AURC gain | Non-regression | Regressions |
|---|---:|---:|---:|---:|
| original_test_split | 0.400 | +9.87% | 90.5% | 4 |
| original_test_split | 0.450 | +9.96% | 100.0% | 0 |
| original_test_split | 0.475 | +9.96% | 100.0% | 0 |
| original_test_split | 0.500 | +9.91% | 100.0% | 0 |
| original_test_split | 0.525 | +9.91% | 100.0% | 0 |
| original_test_split | 0.550 | +9.91% | 100.0% | 0 |
| original_test_split | 0.600 | +9.80% | 100.0% | 0 |
| heldout_class | 0.400 | +8.20% | 84.0% | 64 |
| heldout_class | 0.450 | +8.32% | 87.8% | 49 |
| heldout_class | 0.475 | +8.40% | 89.8% | 41 |
| heldout_class | 0.500 | +8.42% | 90.8% | 37 |
| heldout_class | 0.525 | +8.40% | 91.5% | 34 |
| heldout_class | 0.550 | +8.36% | 92.0% | 32 |
| heldout_class | 0.600 | +8.26% | 92.2% | 31 |

## Gate transfer and lower-bound checks

| Held-out WebFG subset | Selected threshold | Held-out PMRC selected | Mean test gain | Regressions |
|---|---:|---:|---:|---:|
| aircraft | 0.42 | 3/3 | +2.83% | 0 |
| bird | 0.50 | 0/6 | +0.00% | 0 |
| car | 0.42 | 3/3 | +3.24% | 0 |

The lower-bound gate selects PMRC in 36/42 runs, improves 36, leaves 6 unchanged, and produces 0 regressions.

## Prototype contamination and concentration

| Run | Corruption | Noisy-clean prototype cosine | Noisy-margin corruption AUROC | Clean-margin corruption AUROC |
|---|---:|---:|---:|---:|
| elr_resnet50_cub_semantic40_seed0 | 40.7% | 0.9847 | 0.8828 | 0.7346 |
| elr_resnet50_cub_semantic40_seed1 | 40.7% | 0.9776 | 0.8249 | 0.9511 |
| elr_resnet50_cub_semantic40_seed2 | 40.7% | 0.9864 | 0.9044 | 0.6435 |
| elr_resnet50_aircraft_semantic40_seed0 | 40.2% | 0.9844 | 0.7658 | 0.9591 |
| elr_resnet50_flowers102_semantic375_seed0 | 37.5% | 0.9413 | 0.8927 | 0.9881 |
| elr_resnet50_flowers102_semantic375_seed1 | 37.5% | 0.9429 | 0.9030 | 0.9611 |
| elr_resnet50_flowers102_semantic375_seed2 | 37.5% | 0.9522 | 0.9372 | 0.8189 |
| elr_resnet18_cifar100n_human_seed0 | 40.2% | 0.9931 | 0.9035 | 0.8178 |

Across these eight runs, prototype cosine versus PMRC gain has Pearson r=-0.63 (p=0.097) and Spearman rho=-0.50 (p=0.207). This exploratory association is dataset-confounded and is not interpreted causally.

### Class-level factor partition

| Factor | Low-group gain | High-group gain | Low minus high | Runs favoring low |
|---|---:|---:|---:|---:|
| concentration | +11.3% | +22.8% | -11.5 pp | 0/8 |
| prototype_fidelity | +11.4% | +20.8% | -9.4 pp | 0/8 |

### Prototype purification intervention

| Condition | Mean PMRC gain | Mean gate AUROC | PMRC selected |
|---|---:|---:|---:|
| all_noisy | +15.73% | 0.7256 | 8/8 |
| loss_filtered | +16.91% | 0.7370 | 8/8 |
| clean_oracle | +31.68% | 0.8531 | 8/8 |

loss_filtered versus all_noisy: +1.18 points (95% CI +0.72 to +1.68; 8/8 runs).

clean_oracle versus loss_filtered: +14.77 points (95% CI +11.18 to +18.44; 8/8 runs).

clean_oracle versus all_noisy: +15.96 points (95% CI +11.90 to +19.66; 8/8 runs).

### WebFG cross-domain open-set proxy

| Known subset | PMRC AURC gain | MSP unknown AUROC | PMRC unknown AUROC | Gate |
|---|---:|---:|---:|---|
| bird | -9.21% | 0.8950 | 0.8688 | msp |
| aircraft | -18.87% | 0.6480 | 0.5928 | pmrc |
| car | -32.80% | 0.9740 | 0.9325 | pmrc |

PMRC is not suitable for open-set claims in the tested cross-domain proxy.

### Strict CUB 160/40 open-set evaluation

Each classifier was trained on 160 known classes only. The other 40 CUB classes were excluded from training, validation, prototype construction, and calibration.

| Seed | AURC gain | MSP unknown AUROC | PMRC unknown AUROC | Gate |
|---:|---:|---:|---:|---|
| 0 | -12.63% | 0.7032 | 0.6764 | pmrc |
| 1 | -4.21% | 0.7027 | 0.7177 | pmrc |
| 2 | +1.68% | 0.7576 | 0.7609 | pmrc |

Across 3 seeds, the mean strict open-set AURC change was -5.05%. This supports a closed-set scope rather than a dedicated open-set detector claim.

## Training-objective transfer

| Objective | Seeds | MSP AURC | PMRC AURC | Relative gain | Improving seeds |
|---|---:|---:|---:|---:|---:|
| CE | 3 | 0.2327 | 0.1914 | +17.97% | 3/3 |
| ELR | 3 | 0.2109 | 0.1769 | +16.22% | 3/3 |
| GCE | 3 | 0.2090 | 0.1725 | +17.47% | 3/3 |

## Stanford Dogs external-domain validation

The protocol and gate threshold were frozen before training. Official test membership was preserved, and exact duplicate train-side images were removed before splitting.

| Seed | MSP AURC | PMRC AURC | Relative gain | Gate AUROC | Gate |
|---:|---:|---:|---:|---:|---|
| 0 | 0.0861 | 0.0614 | +28.63% | 0.8526 | pmrc |
| 1 | 0.0799 | 0.0588 | +26.46% | 0.7854 | pmrc |
| 2 | 0.0897 | 0.0627 | +30.11% | 0.8162 | pmrc |

Mean relative AURC gain is +28.40%; all 3 completed seeds improve and the gate direction matches in 3/3 runs.

Stanford Dogs shares ImageNet lineage with the pretrained backbone. This validates downstream dataset and split transport, not a pretraining-independent real-web domain.

### Cost and calibration stability

| Run | Single view | Two view ratio | Total post-hoc |
|---|---:|---:|---:|
| elr_resnet50_cub_semantic40_seed0 | 21.1s | 2.48x | 93.2s |
| elr_resnet50_aircraft_semantic40_seed0 | 11.9s | 2.57x | 76.9s |
| elr_resnet50_flowers102_semantic375_seed0 | 21.8s | 2.51x | 67.8s |
| elr_resnet18_cifar100n_human_seed0 | 5.9s | 1.74x | 59.3s |

Across 42 runs, the median relative span between best and worst candidate CV AURC was 2.4%.

## Decision cost

Error cost is normalized to one. Thresholds are selected on
calibration data and evaluated on fixed test data.

| Rejection cost | MSP cost | PMRC cost | PMRC lower | MSP coverage | PMRC coverage |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.0848 | 0.0808 | 35/42 | 29.1% | 33.2% |
| 0.25 | 0.1810 | 0.1709 | 38/42 | 46.0% | 48.9% |
| 0.50 | 0.2886 | 0.2740 | 38/42 | 72.1% | 71.4% |
| 1.00 | 0.3324 | 0.3322 | 26/42 | 99.7% | 99.7% |

## Artifact files

- `calibration_robustness.{json,csv}`
- `heldout_class_calibration.{json,csv}`
- `risk_control_baseline.{json,csv}`
- `risk_constrained_coverage.{json,csv}`
- `gate_prospective_checks.{json,md}`
- `gate_threshold_sensitivity.{json,csv}`
- `prototype_contamination.{json,csv,md}`
- `prototype_gain_association.{json,md}`
- `prototype_factor_partition.{json,csv,md}`
- `prototype_purification_intervention.{json,csv,md}`
- `training_objective_transfer.{json,csv,md}`
- `stanford_dogs_external_validation.{json,csv,md}`
- `openset_webfg_proxy.{json,csv,md}`
- `openset_cub160.{json,csv}`
- `openset_cub160_summary.{json,csv,md}`
- `posthoc_cost_benchmark.{json,csv,md}`
- `cv_stability_summary.{json,md}`
- `decision_cost_analysis.{json,csv}`
