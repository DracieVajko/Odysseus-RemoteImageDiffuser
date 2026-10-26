# Installs NDK r27 + CMake 3.22.1 (PIN in app/build.gradle.kts).
# Long download (~1 GB) — run it in the background.
$ErrorActionPreference = "Continue"
$sdk = "$env:LOCALAPPDATA\Android\Sdk"
$sm = "$sdk\cmdline-tools\latest\bin\sdkmanager.bat"
$yes = Join-Path ([IO.Path]::GetTempPath()) "android_yes.txt"
"y`ny`ny`ny`ny`ny`ny`ny`ny`ny`n" | Out-File -Encoding ascii $yes
Get-Content $yes | & $sm --licenses 2>&1 | Select-Object -Last 2
& $sm "ndk;27.3.13750724" "cmake;3.22.1" 2>&1 | Select-Object -Last 5
Write-Output "NDK-INSTALL-DONE"
ls "$sdk\ndk" 2>$null
ls "$sdk\cmake" 2>$null
