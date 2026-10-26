<#
.SYNOPSIS
Fixes the Windows Firewall rule for Odysseus Diffusion on port 8101.
Run as Administrator.

.DESCRIPTION
Removes the existing incomplete rule and creates a proper one with:
- LocalPort: 8101
- Protocol: TCP
- RemoteAddress: 100.64.0.0/10 (Tailscale CIDR)
- Direction: Inbound
- Action: Allow
#>

param()

if (-NOT ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator!"
    Write-Host "Right-click PowerShell and select 'Run as Administrator', then run this script again."
    exit 1
}

$ErrorActionPreference = "Stop"

Write-Host "Fixing firewall rule for Odysseus Diffusion on port 8101..." -ForegroundColor Cyan

# Remove existing incomplete rule
Remove-NetFirewallRule -DisplayName "Odysseus Diffusion 8101" -ErrorAction SilentlyContinue

# Remove legacy 8100 rule if it exists
Remove-NetFirewallRule -DisplayName "Odysseus Diffusion 8100 (Tailscale only)" -ErrorAction SilentlyContinue

# Create correct rule
New-NetFirewallRule `
    -DisplayName "Odysseus Diffusion 8101" `
    -Description "Allows inbound Tailscale traffic to Odysseus Diffusion server on port 8101" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 8101 `
    -RemoteAddress 100.64.0.0/10 `
    -Action Allow `
    -Enabled True `
    -Profile Domain,Private,Public

Write-Host "Firewall rule created successfully!" -ForegroundColor Green

# Verify
$rule = Get-NetFirewallRule -DisplayName "Odysseus Diffusion 8101"
$rule | Select-Object DisplayName, Enabled, Direction, Action, LocalPort, RemoteAddress, Protocol, Profile | Format-List

Write-Host "`nDone! Port 8101 is now open for Tailscale traffic (100.64.0.0/10)." -ForegroundColor Green