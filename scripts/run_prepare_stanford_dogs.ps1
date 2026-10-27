$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "..\.venv\Scripts\python.exe"
$logRoot = Join-Path $projectRoot "runs\e0_logs"
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

Push-Location $projectRoot
try {
    & $python scripts\prepare_stanford_dogs.py `
        *>> (Join-Path $logRoot "stanford_dogs_prepare.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Stanford Dogs preparation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
