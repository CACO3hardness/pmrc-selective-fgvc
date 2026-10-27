$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "..\.venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "runs\e0_logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

Push-Location $projectRoot
try {
    foreach ($seed in 0, 1, 2) {
        $config = Join-Path $projectRoot "configs\cub_gce_semantic40_seed$seed.yaml"
        $tag = "cub_gce_semantic40_seed$seed"
        $queueLog = Join-Path $logRoot "${tag}_queue.log"
        "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') queued $tag =====" |
            Out-File -LiteralPath $queueLog -Append -Encoding utf8
        & $python -m rgel.seed_pipeline `
            --config $config `
            --seed $seed `
            --tag $tag *>> $queueLog
        if ($LASTEXITCODE -ne 0) {
            throw "$tag failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
