$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$statusPath = Join-Path $projectRoot "runs\stanford_dogs_elr_semantic40_seed2_pipeline_status.json"
$logPath = Join-Path $projectRoot "runs\e0_logs\stanford_dogs_watch.log"
$deadline = (Get-Date).AddHours(6)

while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $statusPath) {
        $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($status.status -eq "complete") {
            "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') seed2 complete =====" |
                Out-File -LiteralPath $logPath -Append -Encoding utf8
            & (Join-Path $PSScriptRoot "run_stanford_dogs_baselines.ps1") `
                *>> $logPath
            exit $LASTEXITCODE
        }
        if ($status.status -eq "failed") {
            "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') seed2 failed =====" |
                Out-File -LiteralPath $logPath -Append -Encoding utf8
            exit 1
        }
    }
    Start-Sleep -Seconds 30
}

throw "Timed out waiting for Stanford Dogs seed2"
