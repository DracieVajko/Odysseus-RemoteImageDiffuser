<#
Install-DiffusionStartup.ps1 - registers a per-user logon task that starts the
diffusion helper in LAZY mode (no model at startup) COMPLETELY HIDDEN.
Runs via silent batch file - NO visible window at all.
Usage:
  .\Install-DiffusionStartup.ps1            # local bind 127.0.0.1
  .\Install-DiffusionStartup.ps1 -Remote    # also reachable via Tailscale
Run .\Uninstall-DiffusionStartup.ps1 to remove.
#>
param([switch]$Remote)

$ErrorActionPreference = "Stop"
$ServerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TaskName = "OdysseusDiffusionHelper"

# Environment for the task
foreach ($kv in @(
  "HF_HOME=D:\AI\HFCache", "HF_HUB_CACHE=D:\AI\HFCache\hub",
  "HUGGINGFACE_HUB_CACHE=D:\AI\HFCache\hub", "HF_HUB_DISABLE_TELEMETRY=1",
  "HF_HUB_VERBOSITY=warning", "TRANSFORMERS_VERBOSITY=warning",
  "TRANSFORMERS_NO_TF=1", "TRANSFORMERS_NO_FLAX=1",
  "PYTHONUTF8=1", "PYTHONUNBUFFERED=1",
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")) {
  $k, $v = $kv.Split("=", 2)
  [Environment]::SetEnvironmentVariable($k, $v, "User")
}

# Use the silent batch file - runs COMPLETELY HIDDEN
$batchPath = Join-Path $ServerDir "run_diffusion_port8101.bat"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$batchPath`"" -WorkingDirectory $ServerDir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -StartWhenAvailable `
  -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -Hidden `
  -RunOnlyIfNetworkAvailable
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

try {
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
  Write-Host "Installed HIDDEN logon task '$TaskName' (user $env:USERNAME, no admin needed)."
  Write-Host "Runs at logon: lazy diffusion server on port 8101 (NO visible window)."
  Write-Host "Start now without reboot: Start-ScheduledTask -TaskName $TaskName"
} catch {
  Write-Error ("Failed to register task: " + $_)
  exit 1
}