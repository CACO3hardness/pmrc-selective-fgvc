# PMRC: Prototype-Margin Reliability Calibration

Code and source data for the manuscript:

**Post-hoc Prototype-Margin Reliability Calibration for Closed-Set Selective
Fine-Grained Recognition under Label Noise**

Xianning Zhao, Haoyuan Tang, and Liang Qiao

## What PMRC Does

PMRC is a post-hoc selective-classification procedure for a frozen
noisy-label classifier. It constructs class prototypes from observed training
labels, combines prototype margin with confidence and two-view consistency,
and fits a correctness ranker on clean calibration labels. A validation-only
gate chooses between PMRC and maximum softmax probability (MSP) ordering.
The classifier prediction is never changed.

The method is evaluated on CUB-200-2011, FGVC-Aircraft, Flowers102,
CIFAR-100N, WebFG-496, and Stanford Dogs.

## Repository Layout

- `rgel/`: training, inference, and reusable metric code.
- `scripts/`: dataset preparation, noise generation, post-hoc calibration,
  baselines, robustness analyses, figure generation, and summary scripts.
- `configs/`: training configurations.
- `tests/`: CPU-safe unit tests.
- `third_party/`: redistributable reference components listed in
  `THIRD_PARTY.md`.
- `source_data/`: source tables used by the paper figures.
- `supplementary_data/`: machine-readable summaries for the supplementary
  experiments.
- `paper/`: Pattern Recognition LaTeX source, bibliography, figure files,
  highlights, cover letter, and submission checklist.

Raw datasets, model checkpoints, cached tensors, and full run directories are
excluded. They can be regenerated with the scripts and public dataset sources
described below.

## Environment

The experiments used Python 3.12, PyTorch with CUDA 12.8, torchvision, and the
packages in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The reported runs used an NVIDIA RTX 5060 Laptop GPU. The post-hoc calibration
is lightweight, but reproducing all training runs requires a CUDA-capable GPU.

## Data

Download the datasets from their official sources:

- CUB-200-2011
- FGVC-Aircraft
- Oxford Flowers102
- CIFAR-100N
- WebFG-496
- Stanford Dogs

The repository does not redistribute source images. Dataset-specific
manifests can be regenerated with `scripts/prepare_cub.py`,
`scripts/prepare_stanford_dogs.py`, and the dataset preparation utilities.
Clean validation/test splits are created before training and are never used
for noisy-label replacement.

## Noise Generation

Semantic-neighbor noise is generated from the top five nearest ImageNet class
centroids. Symmetric noise is generated as a separate control:

```powershell
.\.venv\Scripts\python.exe scripts\make_noise.py `
  --root data\CUB_200_2011 `
  --manifest data\manifests\cub_seed0.json `
  --out data\noise\cub_semantic_40_seed0.csv `
  --metadata-out data\noise\cub_semantic_40_seed0.json `
  --kind semantic `
  --top-k 5 `
  --rate 0.4 `
  --seed 0
```

The training code reads the generated noisy labels. Clean benchmark labels are
not exposed to the classifier during noisy training.

## Training

Run a single configuration with:

```powershell
.\.venv\Scripts\python.exe -m rgel.train --config configs\cub_e0_semantic40.yaml
```

The full queued suites are available through:

```powershell
.\.venv\Scripts\python.exe -m rgel.suite
```

## PMRC Calibration

Post-hoc reliability calibration is implemented in:

```text
scripts/posthoc_reliability.py
```

Key summaries can be regenerated with:

```powershell
.\.venv\Scripts\python.exe scripts\summarize_results.py
.\.venv\Scripts\python.exe scripts\applicability_gate.py
```

## Figures

The vector figures in `paper/figures/` are generated from Python:

```powershell
.\.venv\Scripts\python.exe scripts\make_main_figure.py
.\.venv\Scripts\python.exe scripts\make_webfg_gate_figure.py
.\.venv\Scripts\python.exe scripts\make_method_overview_diagram_v2.py
```

The method overview is a self-contained HTML file with inline SVG. Its PDF is
converted from that source and is used by the manuscript.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The core unit tests do not require the image datasets.

## Manuscript

From `paper/`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\compile_pr.ps1"
```

The Pattern Recognition manuscript uses the locally supplied
`elsarticle.cls` and `elsarticle-num.bst`.

## Scope

The empirical claims are limited to closed-set selective classification with
clean calibration data that covers the observed classes. The gate is a
calibration-side decision rule. It is not an open-set detector, and the
experiments do not establish universal superiority on WebFG or compatibility
with every training-side noisy-label method.

## License

The original PMRC code is released under the MIT License. Files copied from
third-party projects retain their original licenses; see `THIRD_PARTY.md`.
