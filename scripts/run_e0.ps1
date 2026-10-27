$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$python = (Resolve-Path "..\.venv\Scripts\python.exe").Path
$logDir = Join-Path $projectRoot "runs\e0_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-E0Run {
    param(
        [string]$Config,
        [string]$RunName
    )

    $resultPath = Join-Path $projectRoot "runs\$RunName\test_metrics.json"
    if (Test-Path $resultPath) {
        Write-Host "Skipping completed run: $RunName"
        return
    }

    Write-Host "Starting: $RunName"
    & $python -m rgel.train --config $Config --run-name $RunName *>&1 |
        Tee-Object -FilePath (Join-Path $logDir "$RunName.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed: $RunName"
    }
}

Invoke-E0Run `
    -Config "configs\cub_e0_clean.yaml" `
    -RunName "ce_resnet50_cub_clean_seed0"
Invoke-E0Run `
    -Config "configs\cub_e0_semantic40.yaml" `
    -RunName "ce_resnet50_cub_semantic40_seed0"
Invoke-E0Run `
    -Config "configs\cub_elr_semantic40.yaml" `
    -RunName "elr_resnet50_cub_semantic40_seed0"

Write-Host "E0 runs finished. Results are under runs\<run-name>\test_metrics.json."

