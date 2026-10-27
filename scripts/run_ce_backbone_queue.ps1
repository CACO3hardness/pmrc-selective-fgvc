$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "..\.venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "runs\e0_logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

Push-Location $projectRoot
try {
    foreach ($seed in 1, 2) {
        $config = Join-Path $projectRoot "configs\cub_ce_semantic40_seed$seed.yaml"
        $tag = "cub_ce_semantic40_seed$seed"
        "===== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') queued $tag =====" |
            Out-File -LiteralPath (Join-Path $logRoot "${tag}_queue.log") `
                -Append -Encoding utf8
        & $python -m rgel.seed_pipeline `
            --config $config `
            --seed $seed `
            --tag $tag *>> (Join-Path $logRoot "${tag}_queue_console.log")
        if ($LASTEXITCODE -ne 0) {
            throw "$tag failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
