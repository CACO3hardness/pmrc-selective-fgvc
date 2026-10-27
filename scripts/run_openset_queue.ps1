$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "..\.venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "runs\e0_logs"
$manifest = Join-Path $projectRoot "data\manifests\cub_openset160_seed0.json"
$noise = Join-Path $projectRoot "data\noise\cub_openset160_semantic40_seed0.csv"
$noiseMetadata = Join-Path $projectRoot "data\noise\cub_openset160_semantic40_seed0.json"
$config = Join-Path $projectRoot "configs\cub_openset160_elr_semantic40_seed0.yaml"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $noise)) {
        & $python scripts\make_noise.py `
            --root data\CUB_200_2011_256 `
            --manifest $manifest `
            --out $noise `
            --metadata-out $noiseMetadata `
            --kind semantic `
            --top-k 5 `
            --rate 0.4 `
            --seed 0 `
            --batch-size 64 `
            --num-workers 0 *>> (Join-Path $logRoot "cub_openset160_noise.log")
        if ($LASTEXITCODE -ne 0) {
            throw "Open-set noise generation failed with exit code $LASTEXITCODE"
        }
    }
    & $python -m rgel.seed_pipeline `
        --config $config `
        --seed 0 `
        --tag "cub_openset160_elr_semantic40_seed0" *>> (Join-Path $logRoot "cub_openset160_queue.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Open-set classifier failed with exit code $LASTEXITCODE"
    }
    & $python scripts\openset_cub_eval.py *>> (Join-Path $logRoot "cub_openset160_eval.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Open-set evaluation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
