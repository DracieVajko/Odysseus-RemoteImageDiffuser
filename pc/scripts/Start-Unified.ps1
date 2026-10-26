<# Start-Unified.ps1 - Unified standalone launcher (images + audio, port 8101).
Lazy by default (no model at startup, VRAM stays free):
  .\Start-Unified.ps1
  .\Start-Unified.ps1 -Remote          # also bind 0.0.0.0 + TrustedHost for Tailscale IP
  .\Start-Unified.ps1 -Shim            # + legacy :4444 shim (stdlib, for old :4444 clients)
  .\Start-Unified.ps1 -NoEngine        # skip diffusion registry init (placeholder images)
#>
param(
  [switch]$Remote,
  [switch]$Shim,
  [switch]$NoEngine,
  [int]$Port = 8101,
  [string]$WhisperUrl = "http://127.0.0.1:4444"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ServerDir = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $ServerDir
$VenvPython = Join-Path $ServerDir ".venv\Scripts\python.exe"
if (!(Test-Path $VenvPython)) { $VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe" }
$ServerPy = "unified_server.py"
$LogsDir = Join-Path $ServerDir "logs"
$RunDir = Join-Path $ServerDir "run"
$PidFile = Join-Path $RunDir "unified.pid"
$ShimPidFile = Join-Path $RunDir "unified-shim.pid"

function Fail($msg) { Write-Error $msg; exit 1 }

if (!(Test-Path $VenvPython)) { Fail "Missing venv python: $VenvPython" }
if (!(Test-Path (Join-Path $ServerDir $ServerPy))) { Fail "Missing server: $ServerPy" }
if (!(Test-Path (Join-Path $ServerDir "models.json"))) { Fail "Missing models.json" }

$StopScript = Join-Path $ScriptDir "Stop-Unified.ps1"
if (Test-Path $StopScript) { & $StopScript }

$BindHost = "127.0.0.1"
if ($Remote) { $BindHost = "0.0.0.0" }

$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:HF_HUB_VERBOSITY = "warning"
$env:TRANSFORMERS_VERBOSITY = "warning"
$env:TRANSFORMERS_NO_TF = "1"
$env:TRANSFORMERS_NO_FLAX = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:UNIFIED_WHISPER_URL = $WhisperUrl
if ($NoEngine) { $env:UNIFIED_NO_ENGINE = "1" }

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogFile = Join-Path $LogsDir ("unified-{0}.log" -f $timestamp)

Write-Host "Starting unified server (LAZY - no model loaded at startup):"
Write-Host "  Bind  : ${BindHost}:${Port}"
Write-Host "  Whisper fallback: $WhisperUrl"
Write-Host "  Log   : $LogFile"

$proc = Start-Process -FilePath $VenvPython -ArgumentList @("-m", "uvicorn", "unified_server:app", "--host", $BindHost, "--port", "$Port") -WorkingDirectory $ServerDir -RedirectStandardOutput $LogFile -RedirectStandardError ($LogFile + ".err") -NoNewWindow -PassThru
$proc.Id | Out-File -FilePath $PidFile -Encoding ascii
"PID $($proc.Id) started, waiting for /health ..."

$deadline = (Get-Date).AddMinutes(10)
$ready = $false
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Seconds 5
  if ($proc.HasExited) {
    Write-Error "Server process exited early (code $($proc.ExitCode)). See log: $LogFile"
    Get-Content $LogFile -Tail 40
    exit 1
  }
  try {
    $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
    if ($h.status -eq "ok") { $ready = $true; break }
  } catch { }
  Write-Host "  ... waiting (log tail):"
  try { Get-Content $LogFile -Tail 3 } catch { }
}
if (-not $ready) { Fail "Server did not become ready within 10 minutes. Log: $LogFile" }

try {
  $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
  if ($conn.OwningProcess -ne $proc.Id) {
    $conn.OwningProcess | Out-File -FilePath $PidFile -Encoding ascii
    Write-Host "Real server PID $($conn.OwningProcess) (launcher was $($proc.Id)) recorded."
  }
} catch { Write-Warning "Could not resolve listener PID: $_" }

if ($Shim) {
  $ShimLog = Join-Path $LogsDir ("unified-shim-{0}.log" -f $timestamp)
  $shim = Start-Process -FilePath $VenvPython -ArgumentList @("unified_shim.py", "--target", "http://127.0.0.1:$Port", "--port", "4444") -WorkingDirectory $ServerDir -RedirectStandardOutput $ShimLog -RedirectStandardError ($ShimLog + ".err") -NoNewWindow -PassThru
  $shim.Id | Out-File -FilePath $ShimPidFile -Encoding ascii
  Write-Host "Legacy shim :4444 -> :$Port (PID $($shim.Id))"
}

$m = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 10
$ids = ($m.data | ForEach-Object { $_.id }) -join ", "
Write-Host "READY: http://127.0.0.1:$Port  models=[$ids] audio=[$($m.audio_models -join ', ')]"
Write-Host "WebUI: http://127.0.0.1:$Port/webui"
Write-Host "Stop : .\Stop-Unified.ps1"
if ($Remote) {
  try {
    $tsIp = ((tailscale ip -4) | Select-Object -First 1 | Out-String).Trim()
    if ($tsIp -match '^\d+\.\d+\.\d+\.\d+$') { Write-Host "Remote URL (Tailscale): http://${tsIp}:${Port}" }
  } catch { }
}
