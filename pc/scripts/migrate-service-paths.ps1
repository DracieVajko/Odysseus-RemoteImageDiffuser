# migrate-service-paths.ps1 - repoint the OdysseusDiffusion service at pc/
#
# Run once as Administrator after the repository was reorganised:
#   powershell -ExecutionPolicy Bypass -File pc/scripts/migrate-service-paths.ps1
#   ... -Restart   # also restart the service now
#
# The service keeps running the legacy diffusion_server.py unless you use
# Switch-Service-ToUnified.ps1 afterwards.
param([switch]$Restart)

$ErrorActionPreference = "Stop"
$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script as Administrator (it changes a Windows service)."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PcDir = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $PcDir
$Nssm = Join-Path $ScriptDir "nssm.exe"
$Service = "OdysseusDiffusion"
$Python = Join-Path $PcDir ".venv\Scripts\python.exe"
if (!(Test-Path $Python)) { $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe" }
if (!(Test-Path $Python)) { throw "Missing venv python: $Python" }
if (!(Test-Path $Nssm)) { throw "Missing nssm.exe: $Nssm" }

Write-Host "== current service config =="
& $Nssm get $Service Application
& $Nssm get $Service AppParameters
& $Nssm get $Service AppDirectory

# 1) The SCM must start nssm from its new location.
& sc.exe config $Service binPath= "`"$Nssm`"" | Out-Null

# 2) nssm parameters: keep the legacy diffusion server but with pc/ paths.
# Optional: set PRISM_ALLOWED_HOST to the Tailscale/LAN address external
# clients use (e.g. 100.x.y.z); nothing machine-specific is stored here.
$allowedHost = $env:PRISM_ALLOWED_HOST
$hostArg = if ($allowedHost) { " --allowed-host $allowedHost" } else { "" }
$svcArgs = "--lazy --models-json `"$PcDir\models.json`" --port 8101 --host 0.0.0.0$hostArg --disable-safety-checker"
& $Nssm set $Service Application $Python | Out-Null
& $Nssm set $Service AppParameters "`"$PcDir\diffusion_server.py`" $svcArgs" | Out-Null
& $Nssm set $Service AppDirectory $PcDir | Out-Null

Write-Host "== new service config =="
& $Nssm get $Service Application
& $Nssm get $Service AppParameters
& $Nssm get $Service AppDirectory

if ($Restart) {
    Write-Host "Restarting $Service ..."
    & $Nssm restart $Service | Out-Null
    Start-Sleep -Seconds 8
    try {
        Invoke-RestMethod "http://127.0.0.1:8101/health" | ConvertTo-Json -Depth 3
    } catch {
        Write-Warning "health check failed: $_"
    }
} else {
    Write-Host "Paths updated. The running process keeps the old paths until restart."
}
