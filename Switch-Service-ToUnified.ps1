<#
Switch-Service-ToUnified.ps1 - prepne Windows službu OdysseusDiffusion
zo starého diffusion_server.py na unified_server.py.

POZOR: mení bežiacu systémovú službu. Spusti len keď chceš unified
(WebUI + audio + modely) namiesto starého image-only servera.
Revert: .\Switch-Service-ToUnified.ps1 -Revert

Predpoklady:
- služba OdysseusDiffusion existuje (nssm.exe v tomto adresári)
- .venv existuje
#>
param(
  [switch]$Revert,
  [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$Dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Nssm = Join-Path $Dir "nssm.exe"
$Service = "OdysseusDiffusion"
$Python = Join-Path $Dir ".venv\Scripts\python.exe"

if (!(Test-Path $Nssm)) { Write-Error "Missing nssm.exe in $Dir"; exit 1 }

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
