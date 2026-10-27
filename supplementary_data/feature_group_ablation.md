# Feature-group ablation

Lower AURC is better. The prototype columns replace only the prototype-
margin descriptor while preserving all other confidence and two-view
descriptors. Negative deltas indicate improvement.

| Dataset | Method | Backbone | Noise | Rate | n | Confidence | +View | +Prototype | All | Prototype gain vs +View |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| aircraft | elr | resnet50 | semantic | 20.0% | 3 | 0.1213 | 0.1160 | 0.1040 | 0.1031 | +0.0120 |
| aircraft | elr | resnet50 | semantic | 40.0% | 3 | 0.2465 | 0.2374 | 0.2180 | 0.2158 | +0.0194 |
| cifar100n | elr | resnet18 | human | 40.2% | 3 | 0.1631 | 0.1595 | 0.1552 | 0.1519 | +0.0043 |
| cub | ce | resnet50 | semantic | 40.0% | 1 | 0.2079 | 0.2080 | 0.1669 | 0.1664 | +0.0411 |
| cub | ce | resnet50 | symmetric | 40.0% | 1 | 0.1851 | 0.1797 | 0.1716 | 0.1689 | +0.0082 |
| cub | elr | resnet18 | semantic | 40.0% | 1 | 0.2633 | 0.2632 | 0.2383 | 0.2369 | +0.0250 |
| cub | elr | resnet50 | clean | 0.0% | 1 | 0.0510 | 0.0492 | 0.0451 | 0.0447 | +0.0041 |
| cub | elr | resnet50 | semantic | 20.0% | 3 | 0.1002 | 0.1001 | 0.0901 | 0.0894 | +0.0100 |
| cub | elr | resnet50 | semantic | 40.0% | 3 | 0.2118 | 0.2081 | 0.1778 | 0.1769 | +0.0303 |
| cub | elr | resnet50 | symmetric | 40.0% | 3 | 0.1595 | 0.1575 | 0.1457 | 0.1448 | +0.0118 |
| flowers102 | elr | resnet50 | semantic | 20.0% | 3 | 0.0723 | 0.0747 | 0.0604 | 0.0615 | +0.0143 |
| flowers102 | elr | resnet50 | semantic | 40.0% | 3 | 0.1502 | 0.1533 | 0.1163 | 0.1184 | +0.0370 |
| webfg_aircraft | elr | resnet50 | web | uncontrolled | 3 | 0.1241 | 0.1235 | 0.1198 | 0.1182 | +0.0037 |
| webfg_bird | elr | resnet50 | web | uncontrolled | 6 | 0.0786 | 0.0794 | 0.0773 | 0.0782 | +0.0021 |
| webfg_car | elr | resnet50 | web | uncontrolled | 3 | 0.0812 | 0.0836 | 0.0778 | 0.0769 | +0.0057 |
