$ErrorActionPreference = "Stop"
$ServerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ServerDir "logs"
$RunDir = Join-Path $ServerDir "run"
$VenvPython = Join-Path $ServerDir ".venv\Scripts\python.exe"
if (!(Test-Path $VenvPython)) { throw "Missing venv python: $VenvPython" }
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogFile = Join-Path $LogDir ("diffusion-lazy-{0}.log" -f $timestamp)

$env:HF_HOME = Join-Path $ServerDir ".hf_cache"
$env:HF_HUB_CACHE = Join-Path $ServerDir ".hf_cache\hub"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ServerDir ".hf_cache\hub"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:HF_HUB_VERBOSITY = "warning"
$env:TRANSFORMERS_VERBOSITY = "warning"
$env:TRANSFORMERS_NO_TF = "1"
$env:TRANSFORMERS_NO_FLAX = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$argList = @(
  (Join-Path $ServerDir "diffusion_server.py")
  "--lazy"
  "--models-json", (Join-Path $ServerDir "models.json")
  "--port", "8101"
  "--host", "0.0.0.0"
  "--disable-safety-checker"
)
try {
  $tsIp = (tailscale ip -4 2>$null) | Select-Object -First 1
  $tsIp = ($tsIp | Out-String).Trim()
  if ($tsIp -match '^\d+\.\d+\.\d+\.\d+$') { $argList += @("--allowed-host", "$tsIp") }
} catch { Write-Warning "tailscale not found, local-only bind (no --allowed-host)" }

$proc = Start-Process -FilePath $VenvPython -ArgumentList $argList -WorkingDirectory $ServerDir `
  -RedirectStandardOutput $LogFile -RedirectStandardError ($LogFile + ".err") `
  -NoNewWindow -PassThru

$proc.Id | Out-File -FilePath (Join-Path $RunDir "diffusion.pid") -Encoding ascii
"lazy" | Out-File -FilePath (Join-Path $RunDir "diffusion.model") -Encoding ascii