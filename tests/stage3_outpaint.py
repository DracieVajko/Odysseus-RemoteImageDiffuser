"""Outpaint-via-inpaint only. Usage: python stage3_outpaint.py <Model> [steps]"""
import base64, io, json, os, subprocess, sys, time, urllib.request
BASE = "http://127.0.0.1:8100"
model = sys.argv[1]
_ALIAS = {"DreamShaper": "DreamShaper_8", "RealisticVision": "RealisticVision_V6_B1", "MeinaMix": "MeinaMix_V11"}
model = _ALIAS.get(model, model)
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 15
cfg = json.load(open(r"D:\AI\DiffusionServer\models.json", encoding="utf-8"))
prof = cfg["models"][model]
outdir = r"D:\AI\DiffusionServer\tests\output"
from PIL import Image, ImageDraw
def gpu():
    try:
        o = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],text=True,timeout=15).strip().splitlines()[0].split(",")
        return {"used":int(o[0]),"total":int(o[1]),"util":int(o[2])}
    except Exception as e:
        return {"err":str(e)}
src = os.path.join(outdir, model+"-txt2img.png")
img = Image.open(src).convert("RGB")
print("vram_idle:", gpu(), flush=True)
OW, OH = 768, 512
base = img.resize((512, 512))
canvas=Image.new("RGB",(OW,OH),(128,128,128)); canvas.paste(base,(0,0))
om=Image.new("L",(OW,OH),0); ImageDraw.Draw(om).rectangle([512,0,OW,OH],fill=255)
bc,bo=io.BytesIO(),io.BytesIO(); canvas.save(bc,format="PNG"); om.save(bo,format="PNG")
body=json.dumps({"model":model,"image":base64.b64encode(bc.getvalue()).decode(),"mask":base64.b64encode(bo.getvalue()).decode(),"prompt":prof["test_prompt"]+", seamless continuation","width":OW,"height":OH,"steps":steps}).encode()
req=urllib.request.Request(BASE+"/v1/images/inpaint",data=body,headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(req,timeout=600) as r: res=json.loads(r.read().decode())
print("outpaint ok in %.1fs server_elapsed=%s vram=%s" % (time.time()-t0, res.get("elapsed"), gpu()),flush=True)
op=os.path.join(outdir,model+"-outpaint.png")
with open(op,"wb") as f: f.write(base64.b64decode(res["image"]))
oimg=Image.open(op)
print("saved",op,os.path.getsize(op),oimg.size,flush=True)
print("DONE",flush=True)
