$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "..\.venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "runs\e0_logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

Push-Location $projectRoot
try {
    foreach ($seed in 1, 2) {
        $runDir = Join-Path $projectRoot "runs\ce_resnet50_cub_semantic40_seed$seed"

        & $python scripts\selective_baselines.py `
            --run-dir $runDir *>> (Join-Path $logRoot "ce_baselines_seed$seed.log")
        if ($LASTEXITCODE -ne 0) { throw "selective baselines failed for CE seed $seed" }

        & $python scripts\selective_embedding_baselines.py `
            --run-dir $runDir *>> (Join-Path $logRoot "ce_embedding_baselines_seed$seed.log")
        if ($LASTEXITCODE -ne 0) { throw "embedding baselines failed for CE seed $seed" }

        & $python scripts\prototype_independence.py `
            --run-dir $runDir `
            --bins 5 `
            --bootstrap 1000 *>> (Join-Path $logRoot "ce_prototype_independence_seed$seed.log")
        if ($LASTEXITCODE -ne 0) { throw "prototype independence failed for CE seed $seed" }
    }
}
finally {
    Pop-Location
}
