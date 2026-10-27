$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "..\.venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "runs\e0_logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

Push-Location $projectRoot
try {
    foreach ($seed in 0, 1, 2) {
        $runDir = Join-Path $projectRoot "runs\gce_resnet50_cub_semantic40_seed$seed"

        & $python scripts\selective_baselines.py `
            --run-dir $runDir *>> (Join-Path $logRoot "gce_baselines_seed$seed.log")
        if ($LASTEXITCODE -ne 0) { throw "selective baselines failed for seed $seed" }

        & $python scripts\selective_embedding_baselines.py `
            --run-dir $runDir *>> (Join-Path $logRoot "gce_embedding_baselines_seed$seed.log")
        if ($LASTEXITCODE -ne 0) { throw "embedding baselines failed for seed $seed" }

        & $python scripts\prototype_independence.py `
            --run-dir $runDir `
            --bins 5 `
            --bootstrap 1000 *>> (Join-Path $logRoot "gce_prototype_independence_seed$seed.log")
        if ($LASTEXITCODE -ne 0) { throw "prototype independence failed for seed $seed" }
    }
}
finally {
    Pop-Location
}
