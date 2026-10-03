# TROUBLESHOOTING

## Generovanie padá s 500 pri loadovaní modelu
Pozri `logs/service-err.log`, riadky `from_single_file (config=...) failed`.
Typické príčiny:
- Chýbajú configy z HF (single-file modely ich potrebujú): skontroluj internet
  a že `HF_HOME` je normálna cesta. Pozor: NSSM `AppEnvironmentExtra` sa musí
  nastavovať po jednom `+VAR=VAL` (nie jeden `;`-spájaný reťazec — ten rozbil
  všetky HF downloady, `install_service_admin.ps1` je už opravený).
- Zlý `single_file_config` alebo repo zmazané z HF.

## Počas loadu prichádza 503 "loading model"
Správne správanie, nie chyba. Cold load trvá minúty; klient má čakať
(`retry_after: 30`) alebo pollovať `/v1/images/progress/{request_id}`.
Nikdy nereštartuj službu uprostred loadu.

## Klient timeoutuje, hoci server nakoniec dopočíta
1024px/high na 6GB karte = ~6–9 min. Riešenie: menší request (768 + medium),
alebo klientsky timeout 300–420 s. Server drží len JEDEN model v pamäti —
prepnutie stojí ďalší cold load.

## 400 namiesto výsledku
`Size 1x1 too small`, `n must be 1..4`, `not valid base64`,
`not a decodable image` — zlý vstup, server je v poriadku, oprav volanie.

## Čierne PNG
Safety checker je defaultne vypnutý (`--disable-safety-checker`). Ak sa čierna
vráti, v logu hľadaj `BLACKED`. Iná možnosť je fp16 numerika na hranových
GPU — zníž rozlíšenie.

## Port obsadený / nič nepočúva
`Get-NetTCPConnection -LocalPort 8101 -State Listen` → má tam byť python.
Ak nie: `Restart-Service OdysseusDiffusion` (admin) a `logs/service-err.log`.
Dve inštancie naraz (služba + manuálny štart) kolidujú — vždy len jedna.

## Windows Defender flagne testovací príkaz (Trojan:Win32/ClickFix)
False positive: heuristika reaguje na vzor powershell+curl+base64.
Testuj cez `warm_test.py` / `generate_image.py` (Python), nie ručnými
base64 one-linermi. Voliteľne exclusion (admin):
`Add-MpPreference -ExclusionPath 'D:\AI\DiffusionServer'`.

## VRAM OOM na 6 GB
SDXL + 1024px bez offloadu neprejde. Over: `performance: auto` má zvoliť `low`
(viz log `Hardware: ...`). Núdzovo: 512px + low, alebo Raisni `cpu_offload`
v profile (pomalšie, prejde). `nvidia-smi` ukáže, čo drží pamäť (pozor:
LM Studio / prehliadač zdieľajú GPU).

## Odysseus vidí modely, ale negeneruje
Skontroluj z RPi: `curl http://<IP>:8101/health` a `/v1/models`
(`missing_models` musí byť prázdne). Endpoint v Odysseus musí končiť `/v1`.
