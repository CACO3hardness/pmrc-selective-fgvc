$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "..\.venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "runs\e0_logs"
$manifest = Join-Path $projectRoot "data\manifests\cub_openset160_seed0.json"
$output = Join-Path $projectRoot "..\..\outputs\openset_cub160.json"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

Push-Location $projectRoot
try {
    foreach ($seed in 0, 1, 2) {
        $noise = Join-Path $projectRoot "data\noise\cub_openset160_semantic40_seed$seed.csv"
        $noiseMetadata = Join-Path $projectRoot "data\noise\cub_openset160_semantic40_seed$seed.json"
        $config = Join-Path $projectRoot "configs\cub_openset160_elr_semantic40_seed$seed.yaml"
        $tag = "cub_openset160_elr_semantic40_seed$seed"

        if (-not (Test-Path -LiteralPath $noise)) {
            & $python scripts\make_noise.py `
                --root data\CUB_200_2011_256 `
                --manifest $manifest `
                --out $noise `
                --metadata-out $noiseMetadata `
                --kind semantic `
                --top-k 5 `
                --rate 0.4 `
                --seed $seed `
                --batch-size 64 `
                --num-workers 0 *>> (Join-Path $logRoot "cub_openset160_noise_seed$seed.log")
            if ($LASTEXITCODE -ne 0) {
                throw "Open-set noise generation failed for seed $seed"
            }
        }

        & $python -m rgel.seed_pipeline `
            --config $config `
            --seed $seed `
            --tag $tag *>> (Join-Path $logRoot "cub_openset160_queue_seed$seed.log")
        if ($LASTEXITCODE -ne 0) {
            throw "Open-set classifier failed for seed $seed"
        }

        & $python scripts\openset_cub_eval.py `
            --run-dir (Join-Path $projectRoot "runs\elr_resnet50_cub_openset160_semantic40_seed$seed") `
            --output $output `
            --append *>> (Join-Path $logRoot "cub_openset160_eval_seed$seed.log")
        if ($LASTEXITCODE -ne 0) {
            throw "Open-set evaluation failed for seed $seed"
        }
    }
}
finally {
    Pop-Location
}
