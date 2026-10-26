<# adb_test.ps1 - end-to-end test of the Android server through adb forward.
   Uses curl.exe (PowerShell 5.1 has no Invoke-RestMethod -Form).
   Usage: powershell -ExecutionPolicy Bypass -File android/tools/adb_test.ps1
#>
$ErrorActionPreference = "Continue"
$Adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$Local = 8199

Write-Host "== devices =="
& $Adb devices

& $Adb forward --remove tcp:$Local 2>$null | Out-Null
& $Adb forward tcp:$Local tcp:8101 | Out-Null

function Try-Get($path) {
  $out = & curl.exe -s --max-time 15 "http://127.0.0.1:$Local$path"
  if ($LASTEXITCODE -ne 0) { Write-Host "FAIL $path (curl exit $LASTEXITCODE)"; return $null }
  return $out
}

Write-Host "`n== /health =="
$h = Try-Get "/health"
if ($h) { $h }

Write-Host "`n== /v1/models =="
$m = Try-Get "/v1/models"
if ($m) { $m }

Write-Host "`n== transcribe (1s tone, whisper small) =="
$wav = Join-Path ([IO.Path]::GetTempPath()) "adb_tone.wav"
python -c "import math,struct,wave; w=wave.open(r'$wav','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); [w.writeframes(struct.pack('<h', int(9000*math.sin(2*3.14159*440*i/16000)))) for i in range(16000)]; w.close()"
$tr = & curl.exe -s --max-time 300 -F "file=@$wav;type=audio/wav" -F "model=whisper-small-q5_1" -F "language=sk" "http://127.0.0.1:$Local/v1/audio/transcriptions"
Write-Host $tr
Remove-Item $wav -ErrorAction SilentlyContinue

Write-Host "`n== caption (expects 503 will-not-load on mobile) =="
$cap = & curl.exe -s --max-time 60 -o NUL -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{\"image\":\"e30=\"}' "http://127.0.0.1:$Local/v1/images/caption"
Write-Host "HTTP $cap"
& curl.exe -s --max-time 60 -X POST -H "Content-Type: application/json" -d '{\"image\":\"e30=\"}' "http://127.0.0.1:$Local/v1/images/caption"
Write-Host ""

Write-Host "`n== logcat (app only) =="
$p = (& $Adb shell pidof eu.odysseus.diffusion).Trim()
if ($p) { & $Adb logcat -d --pid=$p | Select-String -Pattern 'ApiServer|Exception|whisper' | Select-Object -Last 15 }

Write-Host "`nDONE."
