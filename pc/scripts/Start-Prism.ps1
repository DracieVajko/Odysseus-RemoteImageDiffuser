# Start-Prism.ps1 - start the PC server and open the WebUI.
#
# Double-click Start-Prism.bat in the repository root, or run:
#   powershell -ExecutionPolicy Bypass -File pc/scripts/Start-Prism.ps1
#
# Options:
#   -Port 8101     preferred port (default 8101)
#   -Remote        bind 0.0.0.0 instead of 127.0.0.1
#   -NoBrowser     do not open the browser
#   -Cpu           force PRISM_ONNX_PROVIDER=cpu for this process
param(
  [int]$Port = 8101,
  [switch]$Remote,
  [switch]$NoBrowser,
  [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PcDir = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $PcDir

function Test-PythonDeps($exe) {
  try {
    & $exe -c "import fastapi, uvicorn, onnxruntime" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
  } catch { return $false }
}

function Find-Python {
  $candidates = @(
    (Join-Path $PcDir ".venv\Scripts\python.exe")
  )
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) { $candidates += $cmd.Source }
  $candidates += (Join-Path $RepoRoot ".venv\Scripts\python.exe")
  foreach ($c in $candidates) {
    if ((Test-Path $c) -and (Test-PythonDeps $c)) { return $c }
  }
  throw "No Python with fastapi/uvicorn/onnxruntime found. Create pc/.venv and pip install -r pc/requirements.txt."
}

function Test-Prism($port) {
  try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/webui" -TimeoutSec 3 -UseBasicParsing
    return ($r.StatusCode -eq 200)
  } catch { return $false }
}

function Test-AnyListener($port) {
  try {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop
    return ($null -ne $c)
  } catch { return $false }
}

$python = Find-Python
$BindHost = if ($Remote) { "0.0.0.0" } else { "127.0.0.1" }

# Pick a free/Prism port. 8101 may still be the legacy service without WebUI.
$chosen = $Port
if (Test-Prism $chosen) {
  Write-Host "Prism is already running on port $chosen."
  if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$chosen/webui" }
  exit 0
}
if (Test-AnyListener $chosen) {
  Write-Warning "Port $chosen is used by another server (legacy OdysseusDiffusion has no WebUI)."
  $chosen = 8102
  while (Test-AnyListener $chosen) { $chosen++ }
  Write-Host "Starting Prism on port $chosen instead."
}

$RunDir = Join-Path $PcDir "run"
$LogsDir = Join-Path $PcDir "logs"
New-Item -ItemType Directory -Force -Path $RunDir, $LogsDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogFile = Join-Path $LogsDir "prism-$stamp.log"

if ($Cpu) { $env:PRISM_ONNX_PROVIDER = "cpu" }
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"

Write-Host "Starting Prism (PC server)..."
Write-Host "  python : $python"
Write-Host "  bind   : ${BindHost}:${chosen}"
Write-Host "  log    : $LogFile"

$proc = Start-Process -FilePath $python `
  -ArgumentList @("-m", "uvicorn", "unified_server:app", "--host", $BindHost, "--port", "$chosen") `
  -WorkingDirectory $PcDir `
  -RedirectStandardOutput $LogFile `
  -RedirectStandardError ($LogFile + ".err") `
  -WindowStyle Hidden -PassThru

$proc.Id | Out-File -FilePath (Join-Path $RunDir "unified.pid") -Encoding ascii

$deadline = (Get-Date).AddSeconds(120)
$ready = $false
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Seconds 2
  if ($proc.HasExited) { break }
  try {
    $h = Invoke-RestMethod -Uri "http://127.0.0.1:$chosen/health" -TimeoutSec 3
    if ($h.status -eq "ok") { $ready = $true; break }
  } catch { }
}

if (-not $ready) {
  Write-Error "Prism did not start. Log tail:"
  if (Test-Path $LogFile) { Get-Content $LogFile -Tail 40 }
  if (Test-Path ($LogFile + ".err")) { Get-Content ($LogFile + ".err") -Tail 40 }
  exit 1
}

$url = "http://127.0.0.1:$chosen/webui"
Write-Host "Prism is running: $url"
Write-Host "Stop it with Stop-Prism.bat (or pc/scripts/Stop-Unified.ps1)."
if (-not $NoBrowser) { Start-Process $url }
