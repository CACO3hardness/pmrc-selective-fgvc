param()

$ErrorActionPreference = "Stop"
$env:MIKTEX_AUTOINSTALL = "1"
$buildDir = Join-Path $PSScriptRoot "build_pr"
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

$binCandidates = @(
    (Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"),
    (Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin"),
    "C:\Program Files\MiKTeX\miktex\bin\x64"
)
$binDir = $binCandidates |
    Where-Object { Test-Path -LiteralPath (Join-Path $_ "pdflatex.exe") } |
    Select-Object -First 1
if (-not $binDir) {
    throw "MiKTeX pdflatex.exe was not found."
}
$pdflatex = Join-Path $binDir "pdflatex.exe"
$bibtex = Join-Path $binDir "bibtex.exe"

function Invoke-Pdflatex([string]$target) {
    & $pdflatex `
        --enable-installer `
        -interaction=nonstopmode `
        -halt-on-error `
        -file-line-error `
        "-output-directory=$buildDir" `
        $target
    if ($LASTEXITCODE -ne 0) {
        throw "$target compilation failed with exit code $LASTEXITCODE."
    }
}

Push-Location $PSScriptRoot
try {
    Invoke-Pdflatex "main_pr.tex"

    & $bibtex (Join-Path $buildDir "main_pr")
    if ($LASTEXITCODE -ne 0) {
        throw "BibTeX failed with exit code $LASTEXITCODE."
    }

    Invoke-Pdflatex "main_pr.tex"
    Invoke-Pdflatex "main_pr.tex"
    Invoke-Pdflatex "supplementary.tex"
    Invoke-Pdflatex "supplementary.tex"

    Copy-Item -LiteralPath (Join-Path $buildDir "main_pr.pdf") `
        -Destination (Join-Path $PSScriptRoot "main_pr.pdf") -Force
    Copy-Item -LiteralPath (Join-Path $buildDir "supplementary.pdf") `
        -Destination (Join-Path $PSScriptRoot "supplementary.pdf") -Force
    Write-Host "Created main_pr.pdf and supplementary.pdf"
}
finally {
    Pop-Location
}
