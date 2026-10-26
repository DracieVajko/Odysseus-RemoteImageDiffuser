<# Stop-Unified.ps1 - stops ONLY the unified server (+ shim) tracked in run\unified*.pid.
Never kills unrelated python processes: verifies command line contains unified_server.py.
#>
$ErrorActionPreference = "SilentlyContinue"
$ServerDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Stop-Tracked($PidFile, $Pattern) {
  if (!(Test-Path $PidFile)) { Write-Host "No PID file $PidFile - not running."; return }
  $pidText = (Get-Content $PidFile -Raw).Trim()
  $targetPid = 0
  if (![int]::TryParse($pidText, [ref]$targetPid)) { Write-Host "Bad PID file. Removing."; Remove-Item $PidFile -Force; return }
  $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
  if ($null -eq $proc) { Write-Host "PID $targetPid not running. Cleaning."; Remove-Item $PidFile -Force -ErrorAction SilentlyContinue; return }
  $ok = $false
  try {
    $wmi = Get-CimInstance Win32_Process -Filter "ProcessId=$targetPid" -ErrorAction Stop
    if ($wmi.CommandLine -match $Pattern) { $ok = $true }
  } catch { }
  if (-not $ok) {
    Write-Warning "PID $targetPid does not look like $Pattern - NOT killing."
    exit 1
  }
  Write-Host "Stopping $Pattern PID $targetPid ..."
  Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 3
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
  Write-Host "Stopped."
}

Stop-Tracked (Join-Path $ServerDir "run\unified.pid") "unified_server"
Stop-Tracked (Join-Path $ServerDir "run\unified-shim.pid") "unified_shim\.py"
Write-Host "Done."
