<# Uninstall-DiffusionStartup.ps1 - removes the per-user logon task. No admin needed. #>
$ErrorActionPreference = "Stop"
$TaskName = "OdysseusDiffusionHelper"
try {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
  Write-Host "Removed task '$TaskName'."
} catch {
  Write-Host "No task '$TaskName' found (or removal failed: $_)."
}
