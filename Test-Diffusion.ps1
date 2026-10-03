<# Test-Diffusion.ps1 - quick status check: PID, model, /health, /v1/models, VRAM/RAM.
Usage: .\Test-Diffusion.ps1 [-Model DreamShaper] [-Port 8101]
#>
param([string]$Model = "", [int]$Port = 8101)
$ServerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $ServerDir "run\diffusion.pid"
$ModelFile = Join-Path $ServerDir "run\diffusion.model"
if (Test-Path $PidFile) { Write-Host "PID: $((Get-Content $PidFile -Raw).Trim())" } else { Write-Host "PID: (none)" }
if (Test-Path $ModelFile) { Write-Host "Model: $((Get-Content $ModelFile -Raw).Trim())" } else { Write-Host "Model: (unknown)" }
try {
  $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
  Write-Host "health: $($h | ConvertTo-Json -Compress)"
} catch { Write-Host "health: FAILED $_" }
try {
  $m = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" -TimeoutSec 5
  Write-Host "models: $($m | ConvertTo-Json -Compress)"
} catch { Write-Host "models: FAILED $_" }
Write-Host "--- GPU ---"
try { nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv } catch { Write-Host "nvidia-smi failed: $_" }
Write-Host "--- RAM ---"
try { Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize, FreePhysicalMemory | Format-Table -AutoSize | Out-String | Write-Host } catch { }
Write-Host "--- latest log ---"
$latest = Get-ChildItem (Join-Path $ServerDir "logs\diffusion-*.log") -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest) { Write-Host $latest.FullName; Get-Content $latest.FullName -Tail 15 } else { Write-Host "(no logs)" }
