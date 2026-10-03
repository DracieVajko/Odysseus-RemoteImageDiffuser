"""Stage 1: full-quality txt2img + stats. Usage: python stage1_gen.py <Model>"""
import base64, json, os, subprocess, sys, time, urllib.request
BASE = "http://127.0.0.1:8100"
model = sys.argv[1]
_ALIAS = {"DreamShaper": "DreamShaper_8", "RealisticVision": "RealisticVision_V6_B1", "MeinaMix": "MeinaMix_V11"}
model = _ALIAS.get(model, model)
cfg = json.load(open(r"D:\AI\DiffusionServer\models.json", encoding="utf-8"))
prof = cfg["models"][model]
outdir = r"D:\AI\DiffusionServer\tests\output"
os.makedirs(outdir, exist_ok=True)
def gpu():
    try:
        o = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=15).strip().splitlines()[0].split(",")
        return {"used":int(o[0]),"total":int(o[1]),"util":int(o[2])}
    except Exception as e:
        return {"err":str(e)}
print("vram_idle:", gpu(), flush=True)
body = json.dumps({"model": model, "prompt": prof["test_prompt"], "size": prof["test_size"], "n": 1}).encode()
req = urllib.request.Request(BASE+"/v1/images/generations", data=body, headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    res = json.loads(r.read().decode())
dt=time.time()-t0
print("vram_after:", gpu(), f"gen_seconds={dt:.1f}", flush=True)
p = os.path.join(outdir, model+"-txt2img.png")
with open(p,"wb") as f: f.write(base64.b64decode(res["data"][0]["b64_json"]))
print("saved", p, os.path.getsize(p), flush=True)
