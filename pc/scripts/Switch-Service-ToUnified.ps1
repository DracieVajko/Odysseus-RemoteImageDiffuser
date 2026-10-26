<#
Switch-Service-ToUnified.ps1 - switches the OdysseusDiffusion Windows service
from the old diffusion_server.py to unified_server.py.

WARNING: changes a running system service. Run it only when you want the
unified server (WebUI + audio + models) instead of the old image-only server.
Revert: .\Switch-Service-ToUnified.ps1 -Revert

Prerequisites:
- the OdysseusDiffusion service exists (nssm.exe in scripts/)
- .venv exists
#>
param(
  [switch]$Revert,
  [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dir = Split-Path -Parent $ScriptDir          # pc/
$RepoRoot = Split-Path -Parent $Dir            # repo root (.venv lives here)
$Nssm = Join-Path $ScriptDir "nssm.exe"
$Service = "OdysseusDiffusion"
$Python = Join-Path $Dir ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) { $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe" }

if (!(Test-Path $Nssm)) { Write-Error "Missing nssm.exe in $ScriptDir"; exit 1 }

function Show-Config {
  Write-Host "Application : $(& $Nssm get $Service Application)"
  Write-Host "Parameters  : $(& $Nssm get $Service AppParameters)"
  Write-Host "Directory   : $(& $Nssm get $Service AppDirectory)"
}

Write-Host "== current service config =="
Show-Config

if ($Revert) {
  $oldArgs = "--lazy --models-json `"$Dir\models.json`" --port 8101 --host 0.0.0.0 --allowed-host 100.88.219.120 --disable-safety-checker"
  if ($WhatIf) { Write-Host "WOULD revert to diffusion_server.py $oldArgs"; exit 0 }
  & $Nssm stop $Service | Out-Null
  & $Nssm set $Service Application $Python | Out-Null
  & $Nssm set $Service AppParameters "`"$Dir\diffusion_server.py`" $oldArgs" | Out-Null
  & $Nssm set $Service AppDirectory $Dir | Out-Null
  & $Nssm start $Service | Out-Null
  Write-Host "Reverted to diffusion_server.py"
  Show-Config
  exit 0
}

if (!(Test-Path $Python)) { Write-Error "Missing venv python: $Python"; exit 1 }
$newArgs = "-m uvicorn unified_server:app --host 0.0.0.0 --port 8101"
if ($WhatIf) { Write-Host "WOULD switch to: $Python $newArgs"; exit 0 }

& $Nssm stop $Service | Out-Null
& $Nssm set $Service Application $Python | Out-Null
& $Nssm set $Service AppParameters $newArgs | Out-Null
& $Nssm set $Service AppDirectory $Dir | Out-Null
& $Nssm start $Service | Out-Null
Start-Sleep -Seconds 10

Write-Host "Switched. Health:"
try { Invoke-RestMethod "http://127.0.0.1:8101/health" | ConvertTo-Json -Depth 3 }
catch { Write-Warning "health check failed: $_" }

Show-Config
