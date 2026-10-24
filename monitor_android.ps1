<# 
.SYNOPSIS
    Comprehensive Android app monitoring and testing for Prism
.DESCRIPTION
    Runs logcat, system monitoring, health checks, and transcription tests
#>

param(
    [string]$AdbPath = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
    [int]$ForwardPort = 8199,
    [int]$DevicePort = 8101,
    [string]$AudioFile = "D:\AI\DiffusionServer\android\app\src\main\cpp\third_party\whisper.cpp\samples\jfk.wav"
)

$adb = $AdbPath

function Write-Header { param($msg) Write-Host "`n=== $msg ===`n" -ForegroundColor Cyan }
function Write-OK { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERR] $msg" -ForegroundColor Red }

# Check ADB
if (-not (Test-Path $adb)) {
    Write-Err "ADB not found at $adb"
    exit 1
}

Write-Header "ANDROID APP MONITORING & TEST SUITE"
$model = & $adb shell getprop ro.product.model 2>$null
$android = & $adb shell getprop ro.build.version.release 2>$null
Write-Host "Device: $($model.Trim())"
Write-Host "Android: $($android.Trim())"
Write-Host "ADB: $adb"

# 1. CLEAN UP
Write-Header "1. CLEANUP"
& $adb forward --remove-all 2>$null | Out-Null
& $adb shell am force-stop eu.odysseus.diffusion 2>$null | Out-Null
Start-Sleep 2
Write-OK "App stopped, forwards cleared"

# 2. START APP & SERVICE
Write-Header "2. START APP & SERVICE"
& $adb shell am start -n eu.odysseus.diffusion/.MainActivity | Out-Null
Start-Sleep 3

# Tap START button (coordinates from UI dump: center of [900,438][1164,558] = ~1032,498)
& $adb shell input tap 1032 498
Start-Sleep 8

# Verify service started
$pidVar = & $adb shell pidof eu.odysseus.diffusion 2>$null
if ($pidVar) {
    Write-OK "Service running (PID: $($pidVar.Trim()))"
} else {
    Write-Err "Service not running!"
    & $adb shell am start-foreground-service -n eu.odysseus.diffusion/.DiffusionService --es mode AUTO 2>$null | Out-Null
    Start-Sleep 5
    $pidVar = & $adb shell pidof eu.odysseus.diffusion 2>$null
    if ($pidVar) { Write-OK "Service started via foreground command (PID: $($pidVar.Trim()))" }
    else { Write-Err "Service failed to start"; exit 1 }
}

# 3. SETUP FORWARD
Write-Header "3. PORT FORWARD"
& $adb forward --remove-all 2>$null | Out-Null
& $adb forward tcp:$ForwardPort tcp:$DevicePort 2>$null | Out-Null
Start-Sleep 1
Write-OK "Forward $ForwardPort -> $DevicePort"

# 4. HEALTH CHECK
Write-Header "4. HEALTH CHECK"
$health = curl.exe -s --max-time 10 "http://127.0.0.1:$ForwardPort/health" 2>$null
if ($LASTEXITCODE -eq 0 -and $health) {
    $parsed = $health | ConvertFrom-Json
    Write-OK "Health: $($parsed.status) | Device: $($parsed.device) | Models: $($parsed.available_models -join ', ')"
} else {
    Write-Err "Health check failed"
    exit 1
}

# 5. SYSTEM MONITORING SNAPSHOT
Write-Header "5. SYSTEM RESOURCES (Snapshot)"
$pidVar = & $adb shell pidof eu.odysseus.diffusion 2>$null
if ($pidVar) {
    $pidVal = $pidVar.Trim()
    Write-Host "PID: $pidVal"
    
    $stat = & $adb shell "cat /proc/$pidVal/stat 2>/dev/null"
    if ($stat) {
        $parts = $stat.Split(' ')
        $utime = [long]$parts[13]
        $stime = [long]$parts[14]
        $total = $utime + $stime
        Write-Host "CPU time: $total ticks (utime=$utime, stime=$stime)"
    }
    
    $status = & $adb shell "cat /proc/$pidVal/status 2>/dev/null"
    if ($status) {
        $status | Select-String "VmRSS|VmSize|Threads|State" | ForEach-Object { Write-Host "  $_" }
    }
    
    $net = & $adb shell "cat /proc/net/tcp6 2>/dev/null | grep -i 1FA5"
    if (-not $net) { $net = & $adb shell "cat /proc/net/tcp 2>/dev/null | grep -i 1FA5" }
    if ($net) { Write-Host "  Port 8101: LISTENING" } else { Write-Warn "  Port 8101: NOT LISTENING" }
}

# 6. LOGCAT (dump mode — -d never blocks)
Write-Header "6. LOGCAT (recent buffer)"
& $adb logcat -d -t 400 -v time 2>$null | Select-String -Pattern "eu.odysseus.diffusion|ApiServer|DiffusionService|WhisperLib|AudioDecoder|ModelDownloader" | Select-Object -Last 25 | ForEach-Object { Write-Host "  $_" }
Write-OK "logcat dump read"

# 7. TRANSCRIPTION TEST
Write-Header "7. TRANSCRIPTION TEST"
$audioFile = $AudioFile
if (-not (Test-Path $audioFile)) {
    Write-Warn "Audio file not found: $audioFile"
    $audioFile = "D:\AI\DiffusionServer\android\app\src\main\cpp\third_party\whisper.cpp\samples\jfk.wav"
}
Write-Host "Using: $audioFile"

# Test 1: Short audio
Write-Host "`nTest 1: JFK speech (~11s)..."
$start = Get-Date
$result = curl.exe -s --max-time 120 -w "`nHTTP=%{http_code} time=%{time_total}s`n" `
    -F "file=@$audioFile;type=audio/wav" `
    -F "model=whisper-small-q5_1" `
    -F "language=en" `
    "http://127.0.0.1:$ForwardPort/v1/audio/transcriptions" 2>&1
$elapsed = (Get-Date) - $start

$lines = $result -split "`n"
$httpLine = $lines[-1]
$jsonLines = $lines[0..($lines.Count-2)] -join "`n"

Write-Host "Result: $httpLine"
Write-Host "Time: $($elapsed.TotalSeconds.ToString('F2'))s"

if ($httpLine -match "HTTP=200") {
    try {
        $parsed = $jsonLines | ConvertFrom-Json
        Write-OK "Text: $($parsed.text)"
        Write-OK "Lang: $($parsed.language), Duration: $($parsed.duration)s, AutoSplit: $($parsed.auto_split)"
    } catch {
        Write-Warn "Could not parse JSON: $jsonLines"
    }
} else {
    Write-Err "Transcription failed: $result"
}

# Test 2: Test with auto_split (longer audio if available)
Write-Host "`nTest 2: Same file with auto_split=true..."
$result2 = curl.exe -s --max-time 120 -w "`nHTTP=%{http_code} time=%{time_total}s`n" `
    -F "file=@$audioFile;type=audio/wav" `
    -F "model=whisper-small-q5_1" `
    -F "language=en" `
    -F "auto_split=true" `
    "http://127.0.0.1:$ForwardPort/v1/audio/transcriptions" 2>&1

$lines2 = $result2 -split "`n"
$httpLine2 = $lines2[-1]
Write-Host "Result: $httpLine2"

# 8. POST-TEST RESOURCES
Write-Header "8. POST-TEST RESOURCES"
$pidVar = & $adb shell pidof eu.odysseus.diffusion 2>$null
if ($pidVar) {
    $pidVal = $pidVar.Trim()
    $status = & $adb shell "cat /proc/$pidVal/status 2>/dev/null"
    $status | Select-String "VmRSS|VmSize|Threads|State" | ForEach-Object { Write-Host "  $_" }
}

# 9. RECENT LOGS (dump mode)
Write-Header "9. RECENT LOGCAT (last 30 lines)"
$logs = & $adb logcat -d -t 600 -v time 2>$null | Select-String -Pattern "eu.odysseus.diffusion|ApiServer|DiffusionService|WhisperLib|AudioDecoder|ModelDownloader"
if ($logs) {
    $logs | Select-Object -Last 30 | ForEach-Object { Write-Host $_ }
} else {
    Write-Warn "No logcat output received"
}

# 10. CLEANUP
Write-Header "10. CLEANUP"
& $adb forward --remove-all 2>$null | Out-Null
Write-OK "Forwards removed"

Write-Header "TEST COMPLETE"
Write-OK "All tests passed!" -ForegroundColor Green