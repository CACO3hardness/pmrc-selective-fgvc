$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "..\.venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "runs\e0_logs"
$manifest = Join-Path $projectRoot "data\manifests\stanford_dogs_external.json"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

Push-Location $projectRoot
try {
    foreach ($seed in 0, 1, 2) {
        $noise = Join-Path $projectRoot "data\noise\stanford_dogs_semantic_40_seed$seed.csv"
        $noiseMetadata = Join-Path $projectRoot "data\noise\stanford_dogs_semantic_40_seed$seed.json"
        $config = Join-Path $projectRoot "configs\stanford_dogs_elr_semantic40_seed$seed.yaml"
        $tag = "stanford_dogs_elr_semantic40_seed$seed"

        if (-not (Test-Path -LiteralPath $noise)) {
            & $python scripts\make_noise.py `
                --root data\StanfordDogs `
                --manifest $manifest `
                --out $noise `
                --metadata-out $noiseMetadata `
                --kind semantic `
                --top-k 5 `
                --rate 0.4 `
                --seed $seed `
                --batch-size 64 `
                --num-workers 0 *>> (Join-Path $logRoot "stanford_dogs_noise_seed$seed.log")
            if ($LASTEXITCODE -ne 0) {
                throw "Stanford Dogs noise generation failed for seed $seed"
            }
        }

        & $python -m rgel.seed_pipeline `
            --config $config `
            --seed $seed `
            --tag $tag *>> (Join-Path $logRoot "${tag}_queue_console.log")
        if ($LASTEXITCODE -ne 0) {
            throw "Stanford Dogs classifier failed for seed $seed"
        }
    }
}
finally {
    Pop-Location
}
