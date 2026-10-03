# TROUBLESHOOTING

## Generation fails with 500 while loading a model
Check `logs/service-err.log` for `from_single_file (config=...) failed`.
Typical causes:
- Missing configs from HF (single-file models need them): check internet
  access and that `HF_HOME` is a sane path. Note: NSSM `AppEnvironmentExtra`
  must be set per-variable (`+VAR=VAL` each) — one `;`-joined string sets
  `HF_HOME` to the whole blob and breaks all Hub downloads
  (`install_service_admin.ps1` already does it right).
- Wrong `single_file_config` or a repo deleted from HF.

## 503 "loading model" during cold load
Correct behavior, not an error. Cold load takes minutes; the client should
wait (`retry_after: 30`) or poll `/v1/images/progress/{request_id}`. Never
restart the service mid-load.

## Client times out although the server finishes later
1024px/high on a 6 GB card ≈ 6–12 min. Fix: smaller request (768 + medium),
or client timeout 300–420 s. The server holds only ONE model in memory —
every switch costs another cold load.

## 400 instead of a result
`Size 1x1 too small`, `n must be 1..4`, `not valid base64`,
`not a decodable image` — bad input, the server is fine, fix the call.

## Black PNGs
The safety checker is disabled by default (`--disable-safety-checker`). If
black returns, search the log for `BLACKED`. Otherwise lower the resolution.

## Port busy / nothing listening
`Get-NetTCPConnection -LocalPort 8101 -State Listen` should show python.
If not: `Restart-Service OdysseusDiffusion` (admin) and read
`logs/service-err.log`. Two instances at once (service + manual start)
collide — always exactly one.

## Windows Defender flags a test command (Trojan:Win32/ClickFix)
False positive: the heuristic reacts to the powershell+curl+base64 pattern.
Test via `warm_test.py` / `generate_image.py` (Python), not hand-made base64
one-liners. Optional exclusion (admin):
`Add-MpPreference -ExclusionPath '<server-dir>'`.

## VRAM OOM on 6 GB
SDXL + 1024px without offload won't fit. Check the `Hardware: ...` log line
for the picked preset (`auto` → `low` here). Fallback: 512px + low, or force
`cpu_offload` in the profile (slower, fits). `nvidia-smi` shows what else
holds memory (note: LM Studio / browsers share the GPU).

## Odysseus sees models but won't generate
From the Odysseus host: `curl http://<SERVER-IP>:8101/health` and `/v1/models`
(`missing_models` must be empty). The Odysseus endpoint must end with `/v1`.
