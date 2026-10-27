$ErrorActionPreference = "Stop"

$taskName = "Codex-FG-Semantic-Noise-E0"
$projectRoot = Split-Path -Parent $PSScriptRoot
$resumeScript = Join-Path $PSScriptRoot "resume_e0.ps1"
$argument = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$resumeScript`""

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 24) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Resume the resumable E0 fine-grained semantic-noise experiments after logon." `
    -Force | Out-Null

Start-ScheduledTask -TaskName $taskName
Write-Host "Started scheduled task: $taskName"

