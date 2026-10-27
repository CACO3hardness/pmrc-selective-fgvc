# Cover letter for Pattern Recognition

Dear Editor,

We submit the manuscript entitled "Post-hoc Prototype-Margin Reliability
Calibration for Closed-Set Selective Fine-Grained Recognition under Label Noise"
for consideration as a regular research article in Pattern Recognition.

The manuscript studies whether label-noise training can damage the ordering used
by a selective classifier even when the classifier keeps a useful full-coverage
accuracy. We introduce PMRC, a classifier-preserving post-hoc calibration method
that combines confidence, two-view consistency, and class prototypes estimated
from observed labels. The method changes only the acceptance order. We also
provide a calibration-side diagnostic that determines when PMRC should be used
and when the system should return to maximum softmax probability.

The work is aligned with the journal's interest in pattern recognition theory,
methodology, and applications. It evaluates public fine-grained recognition
datasets, controlled and real label noise, several selective-classification
baselines, and explicit applicability boundaries. Closest related work includes
the selective-classification gap analysis of Rabanser and Papernot (2025), the
post-hoc selective calibration evaluation of Cattelan and Silva (2024), Trust
Score by Jiang et al. (2018), ReCalMatch by Hong et al. (2026), and CYFLOD by
Gilal et al. (2025).

The study uses only public datasets: CUB-200-2011, FGVC-Aircraft,
Flowers102, CIFAR-100N, WebFG-496, and Stanford Dogs. The primary selective
metrics are AURC and selective risk, complemented by accuracy, macro-F1, NLL,
Brier score, ECE, coverage under target risks, unknown-class AUROC for the
open-set stress test, and decision cost.

The manuscript reports negative and boundary results, including the WebFG Bird
fallback regime and the strict closed-set open-set test. It does not claim
universal superiority or open-set rejection capability. We believe the resulting
conditional contribution provides a useful reliability-auditing framework for
pattern-recognition systems trained with imperfect supervision.

The manuscript is original, is not under consideration elsewhere, and has been
approved by all authors. The authors declare no competing interests.

Sincerely,

Xianning Zhao

School of Information Science and Engineering, Lanzhou University,
Lanzhou 730000, Gansu, China

zhxianning2025@lzu.edu.cn
