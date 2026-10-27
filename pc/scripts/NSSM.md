# nssm.exe (Windows service helper)

`nssm.exe` is the bundled [NSSM](https://nssm.cc/) service manager used by
`install_service_admin.ps1`, `Switch-Service-ToUnified.ps1` and
`migrate-service-paths.ps1`. It is committed on purpose: the service scripts
must work offline, and the upstream site (`nssm.cc`) is frequently unavailable.

| Property | Value |
|---|---|
| Version | 2.24 (x64) |
| License | Public domain (Iain Patterson, 2003-2014) |
| Size | 337 920 B |
| SHA256 | `f689ee9af94b00e9e3f0bb072b34caaf207f32dcb4f5782fc9ca351df9a06c97` |

Verify before running the service scripts:

```powershell
(Get-FileHash pc\scripts\nssm.exe -Algorithm SHA256).Hash.ToLower()
```

If the hash does not match, do not run the scripts: replace the binary with a
trusted copy of NSSM 2.24.
