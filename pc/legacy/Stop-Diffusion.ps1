<# Stop-Diffusion.ps1 - stops ONLY the diffusion helper tracked in run\diffusion.pid.
Never kills WanGP or unrelated python processes: verifies command line contains diffusion_server.py.
#>
$ErrorActionPreference = "SilentlyContinue"
$ServerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $ServerDir "run\diffusion.pid"
$ModelFile = Join-Path $ServerDir "run\diffusion.model"

if (!(Test-Path $PidFile)) { Write-Host "No PID file - helper not running (or not started via Start-Diffusion.ps1)."; exit 0 }
$pidText = (Get-Content $PidFile -Raw).Trim()
$targetPid = 0
if (![int]::TryParse($pidText, [ref]$targetPid)) { Write-Host "Bad PID file content. Removing."; Remove-Item $PidFile -Force; exit 0 }

$proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
if ($null -eq $proc) { Write-Host "PID $targetPid not running. Cleaning PID file."; Remove-Item $PidFile -Force -ErrorAction SilentlyContinue; exit 0 }

# Verify it is really our diffusion server.
$ok = $false
try {
  $wmi = Get-CimInstance Win32_Process -Filter "ProcessId=$targetPid" -ErrorAction Stop
  if ($wmi.CommandLine -match "diffusion_server\.py") { $ok = $true }
} catch { }

if (-not $ok) {
  Write-Warning "PID $targetPid does not look like diffusion_server.py - NOT killing. CommandLine check failed."
  Write-Warning "Remove run\diffusion.pid manually if stale."
  exit 1
}

Write-Host "Stopping diffusion helper PID $targetPid ..."
# Kill child processes first (venv python.exe is a launcher stub: the real
# server runs as its child with a different PID).
try {
  $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$targetPid" -ErrorAction Stop
  foreach ($c in $children) {
    if ($c.CommandLine -match "diffusion_server\.py") {
      Write-Host "Stopping child server PID $($c.ProcessId) ..."
      Stop-Process -Id $c.ProcessId -Force -ErrorAction SilentlyContinue
    }
  }
} catch { }
Start-Sleep -Seconds 3
Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
# Fallback: any remaining diffusion_server.py owned by this setup (verified by
# command line, never blind kill). Catches orphaned grandchildren.
try {
  $left = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop |
    Where-Object { $_.CommandLine -match "diffusion_server\.py" -and $_.CommandLine -match "D:\\AI\\DiffusionServer" }
  foreach ($p in $left) {
    Write-Host "Stopping orphaned helper PID $($p.ProcessId) ..."
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
  }
} catch { }
Start-Sleep -Seconds 3
# Wait for VRAM release (nvidia-smi compute apps gone for this pid or port free).
for ($i=0; $i -lt 6; $i++) {
  $still = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
  if ($null -eq $still) { break }
  Start-Sleep -Seconds 2
}
Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
Remove-Item $ModelFile -Force -ErrorAction SilentlyContinue
Write-Host "Stopped. Waiting 5s for VRAM release ..."
Start-Sleep -Seconds 5
try { nvidia-smi --query-compute-apps=pid --format=csv,noheader } catch { }
Write-Host "Done."
