# Fetch the prebuilt ncnn Android Vulkan library (shared) and stage the .so
# for the APK. Requires the shared package: glslang is linked inside libncnn.so,
# which avoids the libc++ symbol mismatch of the static package with NDK 27.
#
# Usage (from android/): powershell -ExecutionPolicy Bypass -File tools/fetch_ncnn.ps1
$ErrorActionPreference = "Stop"
$ver = "20261020"
$root = Split-Path -Parent $PSScriptRoot
$third = Join-Path $root "app\src\main\cpp\third_party"
$zip = Join-Path $env:TEMP "ncnn-$ver-android-vulkan-shared.zip"
$url = "https://github.com/Tencent/ncnn/releases/download/$ver/ncnn-$ver-android-vulkan-shared.zip"
$pkg = Join-Path $third "ncnn-shared\ncnn-$ver-android-vulkan-shared"

if (!(Test-Path (Join-Path $pkg "arm64-v8a\lib\libncnn.so"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $pkg) | Out-Null
    if (!(Test-Path $zip) -or (Get-Item $zip).Length -lt 30644273) {
        for ($i = 0; $i -lt 12; $i++) {
            & curl.exe -sS -L -C - -o $zip $url
            if ((Get-Item $zip).Length -ge 30644273) { break }
            Start-Sleep 2
        }
    }
    Expand-Archive -Path $zip -DestinationPath (Split-Path $pkg) -Force
}

$src = Join-Path $pkg "arm64-v8a\lib\libncnn.so"
if (!(Test-Path $src)) { throw "libncnn.so missing after extract" }

$jni = Join-Path $third "ncnn-jni\arm64-v8a"
New-Item -ItemType Directory -Force -Path $jni | Out-Null
Copy-Item $src (Join-Path $jni "libncnn.so") -Force
Write-Host "ncnn ready: $jni\libncnn.so"
