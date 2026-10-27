# Manifest generation

Large image manifests are generated from the official dataset releases rather
than committed to this repository.

The paper uses fixed seed-specific splits. Regenerate a CUB manifest with:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_cub.py `
  --root data\CUB_200_2011 `
  --out data\manifests\cub_seed0.json `
  --val-fraction 0.1 `
  --seed 0
```

Stanford Dogs uses the protocol implemented by
`scripts/prepare_stanford_dogs.py`; WebFG uses the official clean
validation/test split recorded in the paper and supplementary material.

The generated manifests are input to noise generation and training. They are
not included here because they contain machine-specific image paths.
