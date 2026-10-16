<# adb_test.ps1 — večerný test mobilu cez adb (jeden príkaz).
Predpoklady: telefón s nainštalovanou APK + USB debugging + povolené dialógové okno.
Lokálny port 8199 (aby nekolidoval s PC :8101).

Použitie:
  powershell -ExecutionPolicy Bypass -File android/adb_test.ps1
#>
$ErrorActionPreference = "Continue"
$Adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
$LocalPort = 8199

Write-Host "== adb devices =="
& $Adb devices

Write-Host "== forward tcp:$LocalPort -> tcp:8101 =="
& $Adb forward tcp:$LocalPort tcp:8101

function Get-Json($url) {
  try {
    return Invoke-RestMethod -Uri $url -TimeoutSec 15
  } catch {
    Write-Host "FAIL $url : $($_.Exception.Message)"
    return $null
  }
}

Write-Host "== /health =="
$h = Get-Json "http://127.0.0.1:$LocalPort/health"
if ($h) { $h | ConvertTo-Json -Depth 4 }

Write-Host "== /v1/models =="
$m = Get-Json "http://127.0.0.1:$LocalPort/v1/models"
if ($m) {
  foreach ($d in $m.data) { Write-Host ("  {0}: available={1}" -f $d.id, $d.available) }
  Write-Host ("missing: {0}" -f ($m.missing_models -join ", "))
}

Write-Host "== transcribe smoke (1s tón) =="
$wav = Join-Path ([IO.Path]::GetTempPath()) "adb_tone.wav"
python -c "import math,struct,wave; w=wave.open(r'$wav','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); [w.writeframes(struct.pack('<h', int(9000*math.sin(2*3.14159*440*i/16000)))) for i in range(16000)]; w.close()"
try {
  $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/v1/audio/transcriptions" `
    -Method Post -TimeoutSec 300 `
    -Form @{ file = Get-Item $wav; model = "small"; language = "sk" }
  $resp | ConvertTo-Json -Depth 3
} catch {
  Write-Host "transcribe FAIL: $($_.Exception.Message)"
}
Remove-Item $wav -ErrorAction SilentlyContinue

Write-Host "== caption (očakávané 503 will-not-load na mobile) =="
try {
  $r = Invoke-WebRequest -Uri "http://127.0.0.1:$LocalPort/v1/images/caption" `
    -Method Post -ContentType "application/json" `
    -Body '{"image":"e30="}' -TimeoutSec 30 -SkipHttpErrorCheck
  Write-Host ("HTTP {0}: {1}" -f $r.StatusCode, ($r.Content | Out-String).Substring(0, [Math]::Min(200, $r.Content.Length)))
} catch {
  Write-Host "caption FAIL: $($_.Exception.Message)"
}

Write-Host "== logcat (odysseus) =="
& $Adb logcat -d | Select-String -Pattern 'odysseus|eu.odysseus|whisper' | Select-Object -Last 15

Write-Host "DONE. Forward ostáva: adb forward --remove tcp:$LocalPort (keď netreba)."
