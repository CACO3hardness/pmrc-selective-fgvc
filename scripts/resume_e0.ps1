$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:PYTHONUNBUFFERED = "1"
$pythonw = (Resolve-Path "..\.venv\Scripts\pythonw.exe").Path

& $pythonw -m rgel.suite
exit $LASTEXITCODE

