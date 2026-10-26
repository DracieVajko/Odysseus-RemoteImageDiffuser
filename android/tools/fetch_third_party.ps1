<# fetch_third_party.ps1 — clones the C++ sources outside git (requires the .gitignore exception).
Run once (from the repo root or android/):
  powershell -ExecutionPolicy Bypass -File android/tools/fetch_third_party.ps1
#>
$ErrorActionPreference = "Stop"
$androidRoot = Split-Path -Parent $PSScriptRoot
$cpp = Join-Path $androidRoot "app\src\main\cpp\third_party"
New-Item -ItemType Directory -Force -Path $cpp | Out-Null
if (!(Test-Path (Join-Path $cpp "whisper.cpp\include\whisper.h"))) {
  git clone --depth 1 --branch v1.9.5 `
    https://github.com/ggerganov/whisper.cpp `
    (Join-Path $cpp "whisper.cpp")
} else {
  Write-Host "whisper.cpp already present, skipping."
}
