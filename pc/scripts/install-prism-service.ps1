<#
.SYNOPSIS
    Install Prism as the single always-on server for Odysseus (image + whisper).

.DESCRIPTION
    Replaces the legacy pair of services (OdysseusDiffusion on 8101 and
    whisper-api on 4444) with one Prism server that serves every endpoint:
    images, whisper/Nemotron transcription, upscale, caption, chat, ffmpeg,
    outputs, jobs and batch.

    Models are NOT loaded at startup. The server registers metadata only and
    loads weights on the first request that needs them, then releases them on
    the idle timeout (UNIFIED_IDLE_UNLOAD_SECONDS, default 900 s).

    Run elevated to install a real Windows service (starts at boot).
    Run unelevated to install a per-user logon task instead.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File pc\scripts\install-prism-service.ps1
#>
[CmdletBinding()]
param(
    [int]$Port = 8101,
    [string]$Python = "C:\Python314\python.exe",
    [switch]$RemoveLegacy,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pcDir = Join-Path $repoRoot "pc"
$nssm = Join-Path $repoRoot "nssm.exe"
$serviceName = "Prism"
$taskName = "Prism"
$env:PRISM_SERVICE_PORT = "$Port"

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-ServerArgs {
    "-m uvicorn unified_server:app --host 0.0.0.0 --port $Port"
}

$isAdmin = Test-Admin
Write-Host "Prism installer"
Write-Host "  repo      : $repoRoot"
Write-Host "  python    : $Python"
Write-Host "  port      : $Port"
Write-Host "  elevated  : $isAdmin"
Write-Host ""

if (-not (Test-Path $Python)) {
    throw "python not found: $Python (pass -Python <path>)"
}
if (-not (Test-Path (Join-Path $pcDir "unified_server.py"))) {
    throw "unified_server.py not found under $pcDir"
}

# ---------------------------------------------------------------- uninstall
if ($Uninstall) {
    if ($isAdmin -and (Test-Path $nssm)) {
        & $nssm stop $serviceName 2>$null | Out-Null
        & $nssm remove $serviceName confirm 2>$null | Out-Null
        Write-Host "service removed"
    }
    schtasks /Query /TN $taskName 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        schtasks /Delete /TN $taskName /F | Out-Null
        Write-Host "logon task removed"
    }
    return
}

# ---------------------------------------------------------------- legacy
$legacy = @("OdysseusDiffusion", "whisper-api")
$legacyFound = foreach ($name in $legacy) {
    $svc = Get-Service -Name $name -ErrorAction SilentlyContinue
    if ($svc) { $svc }
}
if ($legacyFound) {
    Write-Host "Legacy services detected:"
    foreach ($svc in $legacyFound) {
        Write-Host ("  {0,-20} status={1,-8} start={2}" -f $svc.Name, $svc.Status, $svc.StartType)
    }
    if ($isAdmin -and $RemoveLegacy) {
        foreach ($name in $legacy) {
            $svc = Get-Service -Name $name -ErrorAction SilentlyContinue
            if (-not $svc) { continue }
            if ($svc.Status -ne "Stopped") {
                Stop-Service -Name $name -Force -ErrorAction SilentlyContinue
            }
            Set-Service -Name $name -StartupType Disabled
            Write-Host "  disabled $name"
        }
    } elseif (-not $isAdmin) {
        Write-Host "  (run elevated with -RemoveLegacy to disable them)"
    }
    Write-Host ""
}

# ---------------------------------------------------------------- install
if ($isAdmin -and (Test-Path $nssm)) {
    $existing = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Updating existing service $serviceName"
        & $nssm stop $serviceName 2>$null | Out-Null
    } else {
        Write-Host "Creating service $serviceName"
        & $nssm install $serviceName $Python | Out-Null
    }
    & $nssm set $serviceName Application $Python | Out-Null
    & $nssm set $serviceName AppParameters (Get-ServerArgs) | Out-Null
    & $nssm set $serviceName AppDirectory $pcDir | Out-Null
    & $nssm set $serviceName DisplayName "Prism unified server (images, whisper, Nemotron)" | Out-Null
    & $nssm set $serviceName Description "One OpenAI-compatible endpoint for Odysseus: image generation, whisper/Nemotron transcription, upscale, caption, chat, ffmpeg. Models load on demand." | Out-Null
    & $nssm set $serviceName Start SERVICE_AUTO_START | Out-Null
    & $nssm set $serviceName AppStdout (Join-Path $pcDir "logs\service-out.log") | Out-Null
    & $nssm set $serviceName AppStderr (Join-Path $pcDir "logs\service-err.log") | Out-Null
    & $nssm set $serviceName AppRotateFiles 1 | Out-Null
    & $nssm set $serviceName AppRotateBytes 10485760 | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $pcDir "logs") | Out-Null

    & $nssm start $serviceName | Out-Null
    Start-Sleep -Seconds 3
    $svc = Get-Service -Name $serviceName
    Write-Host "  service state: $($svc.Status)"
} else {
    # Neither a service nor a scheduled task can be created without elevation,
    # so fall back to the per-user Startup folder. It runs at logon, hidden.
    Write-Host "Not elevated: using the per-user Startup folder instead of a service."
    $startup = [Environment]::GetFolderPath("Startup")
    $runner = Join-Path $pcDir "scripts\start-prism-autostart.ps1"
    $vbs = Join-Path $startup "Prism.vbs"
    $psArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runner`""
    $vbsBody = @"
' Prism autostart (generated by install-prism-service.ps1)
Set sh = CreateObject("WScript.Shell")
sh.Run "powershell.exe $psArgs", 0, False
"@
    Set-Content -Path $vbs -Value $vbsBody -Encoding ASCII
    Write-Host "  wrote $vbs"

    # Start it now as well so the endpoint is live immediately.
    Start-Process -FilePath "powershell.exe" -ArgumentList $psArgs -WindowStyle Hidden
    Start-Sleep -Seconds 3
    Write-Host "  started hidden runner"
}

# ---------------------------------------------------------------- verify
$deadline = (Get-Date).AddSeconds(60)
$ok = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
        $ok = $true
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $ok) {
    Write-Warning "Server did not answer on port $Port within 60 s. Check the logs."
    return
}

Write-Host ""
Write-Host "Health: $($r.status)"
Write-Host "  models: $($r.available_models.Count) registered, none loaded yet"
Write-Host "  busy  : $($r.busy)"
Write-Host ""
Write-Host "Endpoints for Odysseus (single base URL):"
Write-Host "  http://<this-host>:$Port/v1"
Write-Host "    POST /v1/images/generations      image generation"
Write-Host "    POST /v1/audio/transcriptions    whisper or nemotron-asr-0.6b"
Write-Host "    POST /v1/chat/completions        VLM chat"
Write-Host "    GET  /health                     readiness probe"
Write-Host ""
Write-Host "Models load on the first request and unload after the idle timeout."
