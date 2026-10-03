# NSSM Service Install Script - Run as Administrator
# Save as install_service_admin.ps1 and run with right-click > "Run with PowerShell" as Admin

# NSSM Service Install Script - Run as Administrator.
# Installs the lazy diffusion server as an auto-start Windows service.
param([string]$TailscaleIp = "")

$ErrorActionPreference = "Stop"

$serverDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$nssm = Join-Path $serverDir "nssm.exe"
$serviceName = "OdysseusDiffusion"
$pythonExe = Join-Path $serverDir ".venv\Scripts\python.exe"
$serverPy = Join-Path $serverDir "diffusion_server.py"
$modelsJson = Join-Path $serverDir "models.json"
$hfCache = Join-Path $serverDir ".hf_cache"

if (!(Test-Path $pythonExe)) { throw "Missing venv python: $pythonExe" }

if (-not $TailscaleIp) {
  try {
    $TailscaleIp = ((tailscale ip -4 2>$null) | Select-Object -First 1 | Out-String).Trim()
  } catch { $TailscaleIp = "" }
}
if ($TailscaleIp -notmatch '^\d+\.\d+\.\d+\.\d+$') {
  Write-Warning "No Tailscale IP (use -TailscaleIp <ip> for remote access); service will be local-only."
  $TailscaleIp = ""
}

# Stop and remove existing
& $nssm stop $serviceName 2>$null
& $nssm remove $serviceName confirm 2>$null

# Install
$args = @($serverPy, "--lazy", "--models-json", $modelsJson, "--port", "8101",
          "--host", "0.0.0.0", "--disable-safety-checker")
if ($TailscaleIp) { $args += @("--allowed-host", $TailscaleIp) }
& $nssm install $serviceName $pythonExe $args

# Configure
& $nssm set $serviceName AppDirectory $serverDir
# NSSM parses AppEnvironmentExtra at the FIRST '=' only: a single ;-joined
# string would set HF_HOME to the whole blob (this broke all HF Hub
# downloads: cache/token paths became garbage). Reset + append each
# VAR=VAL as its own entry instead.
& $nssm reset $serviceName AppEnvironmentExtra
foreach ($kv in @(
  "HF_HOME=$hfCache", "HF_HUB_CACHE=$hfCache\hub",
  "HUGGINGFACE_HUB_CACHE=$hfCache\hub", "HF_HUB_DISABLE_TELEMETRY=1",
  "HF_HUB_VERBOSITY=warning", "TRANSFORMERS_VERBOSITY=warning",
  "TRANSFORMERS_NO_TF=1", "TRANSFORMERS_NO_FLAX=1",
  "PYTHONUTF8=1", "PYTHONUNBUFFERED=1",
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")) {
  & $nssm set $serviceName AppEnvironmentExtra "+$kv"
}
& $nssm set $serviceName Start SERVICE_AUTO_START
& $nssm set $serviceName Type SERVICE_WIN32_OWN_PROCESS
& $nssm set $serviceName ObjectName "NT AUTHORITY\SYSTEM"
& $nssm set $serviceName Description "Odysseus Diffusion Helper - Lazy loading SD server on port 8101"

# Start
& $nssm start $serviceName

Write-Host "Service $serviceName installed and started"