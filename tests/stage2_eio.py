"""Stage 2: edits + inpaint + outpaint-via-inpaint. Usage: python stage2_eio.py <Model>"""
import base64, io, json, os, subprocess, sys, time, urllib.request
BASE = "http://127.0.0.1:8100"
model = sys.argv[1]
_ALIAS = {"DreamShaper": "DreamShaper_8", "RealisticVision": "RealisticVision_V6_B1", "MeinaMix": "MeinaMix_V11"}
model = _ALIAS.get(model, model)
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
def post_json(path, obj, timeout=600):
    body = json.dumps(obj).encode()
    req = urllib.request.Request(BASE+path, data=body, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())
src = os.path.join(outdir, model+"-txt2img.png")
img = Image.open(src).convert("RGB")
w,h = img.size
print("src:", src, img.size, flush=True)
# --- edits (multipart) ---
bnd="----eio123"
with open(src,"rb") as f: png=f.read()
b=io.BytesIO()
for k,v in {"prompt":"make it slightly more vibrant, same composition","model":model,"size":f"{w}x{h}"}.items():
    b.write(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
b.write(f"--{bnd}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"in.png\"\r\nContent-Type: image/png\r\n\r\n".encode())
b.write(png+f"\r\n--{bnd}--\r\n".encode())
t0=time.time()
req=urllib.request.Request(BASE+"/v1/images/edits",data=b.getvalue(),headers={"Content-Type":f"multipart/form-data; boundary={bnd}"})
import urllib.error as _ue
try:
    with urllib.request.urlopen(req,timeout=600) as r: er=json.loads(r.read().decode())
    print(f"edit ok in {time.time()-t0:.1f}s vram={gpu()}", flush=True)
    ep=os.path.join(outdir,model+"-edit.png")
    with open(ep,"wb") as f: f.write(base64.b64decode(er["data"][0]["b64_json"]))
    print("saved",ep,os.path.getsize(ep),flush=True)
except _ue.HTTPError as e:
    body=e.read().decode()[:300]
    print(f"edit HTTP {e.code} (expected for txt2img-only pipeline): {body}", flush=True)
    assert e.code==400 and "does not support image edits" in body, body
# --- inpaint (center mask, capped at 512 working size: 6GB VRAM thrashes at 768 inpaint) ---
IW, IH = 512, 512
ri = img.resize((IW, IH))
mask=Image.new("L",(IW,IH),0); ImageDraw.Draw(mask).rectangle([IW//4,IH//4,3*IW//4,3*IH//4],fill=255)
bi,bm=io.BytesIO(),io.BytesIO(); ri.save(bi,format="PNG"); mask.save(bm,format="PNG")
t0=time.time()
ir=post_json("/v1/images/inpaint",{"model":model,"image":base64.b64encode(bi.getvalue()).decode(),"mask":base64.b64encode(bm.getvalue()).decode(),"prompt":"a beautiful garden fountain, detailed","width":IW,"height":IH})
print(f"inpaint ok in {time.time()-t0:.1f}s server_elapsed={ir.get('elapsed')} vram={gpu()}",flush=True)
ip=os.path.join(outdir,model+"-inpaint.png")
with open(ip,"wb") as f: f.write(base64.b64decode(ir["image"]))
print("saved",ip,os.path.getsize(ip),flush=True)
# --- outpaint via inpaint (512 source -> 768 wide canvas) ---
OW, OH = 768, 512
base = img.resize((512, 512))
canvas=Image.new("RGB",(OW,OH),(128,128,128)); canvas.paste(base,(0,0))
om=Image.new("L",(OW,OH),0); ImageDraw.Draw(om).rectangle([512,0,OW,OH],fill=255)
bc,bo=io.BytesIO(),io.BytesIO(); canvas.save(bc,format="PNG"); om.save(bo,format="PNG")
t0=time.time()
or_=post_json("/v1/images/inpaint",{"model":model,"image":base64.b64encode(bc.getvalue()).decode(),"mask":base64.b64encode(bo.getvalue()).decode(),"prompt":prof["test_prompt"]+", seamless continuation","width":OW,"height":OH})
print(f"outpaint ok in {time.time()-t0:.1f}s server_elapsed={or_.get('elapsed')} vram={gpu()}",flush=True)
op=os.path.join(outdir,model+"-outpaint.png")
with open(op,"wb") as f: f.write(base64.b64decode(or_["image"]))
oimg=Image.open(op)
print("saved",op,os.path.getsize(op),oimg.size,flush=True)
print("DONE",flush=True)
