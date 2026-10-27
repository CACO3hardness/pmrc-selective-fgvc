$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "..\.venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "runs\e0_logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

Push-Location $projectRoot
try {
    foreach ($seed in 0, 1, 2) {
        $runDir = Join-Path $projectRoot "runs\elr_resnet50_stanford_dogs_semantic40_seed$seed"

        & $python scripts\selective_baselines.py `
            --run-dir $runDir *>> (Join-Path $logRoot "stanford_dogs_baselines_seed$seed.log")
        if ($LASTEXITCODE -ne 0) {
            throw "selective baselines failed for Stanford Dogs seed $seed"
        }

        & $python scripts\selective_embedding_baselines.py `
            --run-dir $runDir *>> (Join-Path $logRoot "stanford_dogs_embedding_baselines_seed$seed.log")
        if ($LASTEXITCODE -ne 0) {
            throw "embedding baselines failed for Stanford Dogs seed $seed"
        }

        & $python scripts\prototype_independence.py `
            --run-dir $runDir `
            --bins 5 `
            --bootstrap 1000 *>> (Join-Path $logRoot "stanford_dogs_prototype_independence_seed$seed.log")
        if ($LASTEXITCODE -ne 0) {
            throw "prototype independence failed for Stanford Dogs seed $seed"
        }
    }

    & $python scripts\summarize_stanford_dogs_external.py `
        *>> (Join-Path $logRoot "stanford_dogs_external_summary.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Stanford Dogs external-domain summary failed"
    }
}
finally {
    Pop-Location
}
