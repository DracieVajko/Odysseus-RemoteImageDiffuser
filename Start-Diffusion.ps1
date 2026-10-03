<# Start-Diffusion.ps1 - Odysseus image helper launcher (port 8101).
Preferred always-running mode (loads NO model at startup, lazy per-request):
  .\Start-Diffusion.ps1 -Lazy
  .\Start-Diffusion.ps1 -Lazy -Remote   # also bind 0.0.0.0 + TrustedHost for Tailscale IP

Legacy immediate-load mode (one model at startup, backwards compatible):
  .\Start-Diffusion.ps1 -Model DreamShaper_8
  .\Start-Diffusion.ps1 -Model RealisticVision_V6_B1 -Remote
Old short names still work: DreamShaper, RealisticVision, MeinaMix.
#>
param(
  [string]$Model = "",
  [switch]$Lazy,
  [switch]$Remote,
  [int]$Port = 8101,
  [switch]$NoWait
)

$ErrorActionPreference = "Stop"
$ServerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigPath = Join-Path $ServerDir "models.json"
$VenvPython = Join-Path $ServerDir ".venv\Scripts\python.exe"
$ServerPy = Join-Path $ServerDir "diffusion_server.py"
$LogsDir = Join-Path $ServerDir "logs"
$RunDir = Join-Path $ServerDir "run"
$PidFile = Join-Path $RunDir "diffusion.pid"
$ModelFile = Join-Path $RunDir "diffusion.model"

function Fail($msg) { Write-Error $msg; exit 1 }

if (!(Test-Path $ConfigPath)) { Fail "Missing $ConfigPath" }
if (!(Test-Path $VenvPython)) { Fail "Missing venv python: $VenvPython" }
if (!(Test-Path $ServerPy)) { Fail "Missing server: $ServerPy" }

$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json

# Legacy short names -> registry friendly IDs.
$AliasMap = @{
  "DreamShaper" = "DreamShaper_8"
  "RealisticVision" = "RealisticVision_V6_B1"
  "MeinaMix" = "MeinaMix_V11"
}
if ($Model -ne "" -and $AliasMap.ContainsKey($Model)) { $Model = $AliasMap[$Model] }

$LazyMode = $Lazy -or ($Model -eq "")
if (-not $LazyMode) {
  $profile = $config.models.$Model
  if ($null -eq $profile) { Fail "Unknown model '$Model'. Available: $(($config.models.PSObject.Properties.Name) -join ', ')" }
  $ckptPath = if ($profile.path) { $profile.path } else { $profile.file }
  if (!(Test-Path $ckptPath)) { Fail "Model file missing: $ckptPath" }
} else {
  # Validate registry paths without loading anything (lazy mode).
  foreach ($p in $config.models.PSObject.Properties) {
    $ckptPath = if ($p.Value.path) { $p.Value.path } else { $p.Value.file }
    if (!(Test-Path $ckptPath)) { Fail "Registry model '$($p.Name)' path missing: $ckptPath" }
  }
}

# Stop previous helper instance ONLY (verified via PID file + command line).
$StopScript = Join-Path $ServerDir "Stop-Diffusion.ps1"
if (Test-Path $StopScript) {
  & $StopScript
}

# Determine bind host.
$BindHost = "127.0.0.1"
$AllowedHosts = @()
if ($Remote) {
  $BindHost = "0.0.0.0"
  try {
    $tsIp = (tailscale ip -4) | Select-Object -First 1
    $tsIp = ($tsIp | Out-String).Trim()
    if ($tsIp -match '^\d+\.\d+\.\d+\.\d+$') { $AllowedHosts += $tsIp }
  } catch { Write-Warning "tailscale ip -4 failed: $_" }
}

# Environment: HF cache + workaround for transformers 5.17 import hang
# (telemetry probe blocks on this network; disabling it makes import succeed).
# HF cache: only override when models.json sets hf_cache (empty = HF defaults).
if ($config.hf_cache -and "$($config.hf_cache)".Trim() -ne "") {
  $env:HF_HOME = $config.hf_cache
  $env:HF_HUB_CACHE = Join-Path $config.hf_cache "hub"
  $env:HUGGINGFACE_HUB_CACHE = Join-Path $config.hf_cache "hub"
}
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:HF_HUB_VERBOSITY = "warning"
$env:TRANSFORMERS_VERBOSITY = "warning"
$env:TRANSFORMERS_NO_TF = "1"
$env:TRANSFORMERS_NO_FLAX = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
# Reduce CUDA fragmentation when VRAM is nearly full (6 GB card, SD inpaint
# otherwise thrashes on cudaMalloc retry). No behavior change otherwise.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogTag = if ($LazyMode) { "lazy" } else { $Model }
$LogFile = Join-Path $LogsDir ("diffusion-{0}-{1}.log" -f $LogTag, $timestamp)

$neg = $config.negative_prompt
$argList = @("`"$ServerPy`"", "--port", "$Port", "--host", "$BindHost")
if ($LazyMode) {
  $argList += @("--lazy", "--models-json", "`"$ConfigPath`"", "--disable-safety-checker")
  if ($config.default_model) { $argList += @("--default-model", "$($config.default_model)") }
} else {
  $ckptPath = if ($profile.path) { $profile.path } else { $profile.file }
  $argList += @(
    "--model", "`"$ckptPath`"",
    "--single-file-config", "`"$($profile.single_file_config)`"",
    "--pipeline", "$($profile.pipeline)",
    "--dtype", "$($profile.dtype)",
    "--steps", "$($profile.steps)",
    "--guidance-scale", "$($profile.guidance_scale)",
    "--width", "$($profile.width)",
    "--height", "$($profile.height)",
    "--negative-prompt", "`"$neg`""
  )
  if ($profile.attention_slicing) { $argList += "--attention-slicing" }
  if ($profile.vae_slicing) { $argList += "--vae-slicing" }
  if ($profile.cpu_offload) { $argList += "--cpu-offload" }
}

if ($LazyMode) {
  Write-Host "Starting diffusion helper (LAZY - no model loaded at startup):"
} else {
  Write-Host "Starting diffusion helper:"
  Write-Host "  Model : $Model ($($profile.label))"
  Write-Host "  File  : $ckptPath"
}
Write-Host "  Bind  : ${BindHost}:${Port}"
if ($AllowedHosts) { Write-Host "  TrustedHost extra: $($AllowedHosts -join ', ')" }
Write-Host "  Log   : $LogFile"
foreach ($h in $AllowedHosts) { $argList += @("--allowed-host", "$h") }

if ($LazyMode) {
  Write-Host "Starting diffusion helper (LAZY - no model loaded at startup):"
} else {
  Write-Host "Starting diffusion helper:"
  Write-Host "  Model : $Model ($($profile.label))"
  Write-Host "  File  : $ckptPath"
}
Write-Host "  Bind  : ${BindHost}:${Port}"
if ($AllowedHosts) { Write-Host "  TrustedHost extra: $($AllowedHosts -join ', ')" }
Write-Host "  Log   : $LogFile"

$proc = Start-Process -FilePath $VenvPython -ArgumentList $argList -WorkingDirectory $ServerDir -RedirectStandardOutput $LogFile -RedirectStandardError ($LogFile + ".err") -NoNewWindow -PassThru
$proc.Id | Out-File -FilePath $PidFile -Encoding ascii
$(if ($LazyMode) { "lazy" } else { $Model }) | Out-File -FilePath $ModelFile -Encoding ascii
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

$m = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 10
$ids = ($m.data | ForEach-Object { $_.id }) -join ", "
Write-Host "READY: http://127.0.0.1:$Port  models=[$ids]"
# The venv python.exe is a launcher stub: the real server is its child with a
# different PID. Resolve it via the listening socket so Stop gets the truth.
try {
  $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
  if ($conn.OwningProcess -ne $proc.Id) {
    $conn.OwningProcess | Out-File -FilePath $PidFile -Encoding ascii
    Write-Host "Real server PID $($conn.OwningProcess) (launcher was $($proc.Id)) recorded."
  }
} catch { Write-Warning "Could not resolve listener PID: $_" }
Write-Host "Stop : .\Stop-Diffusion.ps1"
Write-Host "Test : .\Test-Diffusion.ps1 -Model $Model"
if ($Remote) {
  $tsShow = ($AllowedHosts | Select-Object -First 1)
  if ($tsShow) { Write-Host "Remote URL (Tailscale): http://${tsShow}:${Port}" }
}
