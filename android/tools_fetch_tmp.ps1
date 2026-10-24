# Downloads Android cmdline-tools into the SDK (prerequisite for NDK/CMake via sdkmanager).
# curl.exe instead of Invoke-WebRequest (the latter fails in a non-interactive shell, 0x5AD).
$ErrorActionPreference = "Stop"
$sdk = "$env:LOCALAPPDATA\Android\Sdk"
$zip = "$env:TEMP\cmdtools.zip"
$url = "https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip"
if (!(Test-Path "$sdk\cmdline-tools\latest\bin\sdkmanager.bat")) {
  & curl.exe -sS -L -C - -o $zip $url
  if ($LASTEXITCODE -ne 0) { throw "curl exit $LASTEXITCODE" }
  $tmp = "$env:TEMP\cmdtools_ex"
  if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
  Expand-Archive -Path $zip -DestinationPath $tmp -Force
  New-Item -ItemType Directory -Force -Path "$sdk\cmdline-tools\latest" | Out-Null
  Copy-Item "$tmp\cmdline-tools\*" "$sdk\cmdline-tools\latest\" -Recurse -Force
  Remove-Item $tmp -Recurse -Force
  Remove-Item $zip -Force
  Write-Output "CMDTOOLS-INSTALLED"
} else {
  Write-Output "CMDTOOLS-PRESENT"
}
& "$sdk\cmdline-tools\latest\bin\sdkmanager.bat" --list 2>&1 | Select-String -Pattern 'ndk;|cmake;' | Select-Object -First 12
