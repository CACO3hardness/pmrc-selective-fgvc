# WebFG bird replication protocol

## Registration timing

This protocol was written before inspecting PMRC results for seeds 3, 4, or 5.
The original seed0-2 results were already known.

## Fixed protocol

- Dataset: WebFG-496 `web-bird`.
- Classifier: ImageNet-pretrained ResNet-50.
- Freeze stages: 1.
- Input size: 224 x 224.
- Batch size: 64.
- Epochs: 12.
- Method: ELR with beta 0.9 and lambda 3.0.
- Learning rate: 1e-4.
- New classifier seeds: 3, 4, and 5.
- Clean calibration/test split: unchanged from seed0-2 and shared across all
  classifier seeds.
- Post-hoc method: the existing ten-descriptor PMRC without modification.
- Applicability threshold: fixed at 0.5 before evaluation.

## Stopping and reporting rule

- Run each of seeds 3, 4, and 5 exactly once.
- Report all three new results regardless of direction or significance.
- Do not replace, discard, or rerun a seed based on its result.
- Do not add a fourth replication seed in response to the first three results.
- Present original seeds 0-2, replication seeds 3-5, and all six seeds
  separately.
- Any interpretation will use all completed seeds and will not select the
  favorable subset.

## Started at

2026-10-26 13:23:51 Asia/Shanghai.
