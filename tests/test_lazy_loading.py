#!/usr/bin/env python3
"""
Lazy loading acceptance test for Odysseus diffusion helper.
Tests the complete lazy loading flow as specified in the requirements.
"""
import os
import sys
import time
import json
import subprocess
import threading
import urllib.request
import urllib.error

SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SERVER_DIR)

def http_json(url, data=None, timeout=600):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def http_get(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())

def nvidia_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip().splitlines()[0].split(",")
        return {"vram_used_mib": int(out[0].strip()), "vram_total_mib": int(out[1].strip()), "gpu_util_pct": int(out[2].strip())}
    except Exception as e:
        return {"vram_error": str(e)}

def run_powershell(args):
    return subprocess.run(["powershell", "-Command", "& " + " ".join(args)], capture_output=True, text=True, cwd=SERVER_DIR)

def main():
    # Start the server in lazy mode
    print("=== Starting diffusion server in lazy mode ===")
    start_result = run_powershell([".\Start-Diffusion.ps1", "-Lazy"])
    if start_result.returncode != 0:
        print(f"Failed to start server: {start_result.stderr}")
        return

    # Wait for server to be ready
    print("Waiting for server to be ready...")
    for i in range(60):  # 5 minutes timeout
        try:
            health = http_get("http://127.0.0.1:8100/health")
            if health.get("status") == "ok":
                print(f"Server ready: {health}")
                break
        except Exception as e:
            if i % 10 == 0:
                print(f"Waiting... ({i*5}s)")
        time.sleep(5)
    else:
        print("Server failed to become ready within 5 minutes")
        return

    try:
        # Test 1: Verify startup without model
        print("\n=== Test 1: Startup without model ===")
        health = http_get("http://127.0.0.1:8100/health")
        print(f"Health: {health}")
        if health.get("model") is not None:
            print("FAIL: Server has a model loaded at startup")
            return

        # Test 2: Verify /v1/models lists all 3 models
        print("\n=== Test 2: /v1/models lists all 3 models ===")
        models = http_get("http://127.0.0.1:8100/v1/models")
        print(f"Models: {json.dumps(models, indent=2)}")
        if len(models.get("data", [])) != 3:
            print(f"FAIL: Expected 3 models, got {len(models.get('data', []))}")
            return

        model_ids = [m["id"] for m in models["data"]]
        expected_ids = ["DreamShaper_8", "RealisticVision_V6_B1", "MeinaMix_V11"]
        for expected in expected_ids:
            if expected not in model_ids:
                print(f"FAIL: Missing model {expected}")
                return
        print("PASS: All 3 models listed")

        # Test 3: Lazy load DreamShaper
        print("\n=== Test 3: Lazy load DreamShaper ===")
        vram_before = nvidia_mem()
        print(f"VRAM before: {vram_before}")

        gen = http_json("http://127.0.0.1:8100/v1/images/generations", {
            "model": "DreamShaper_8",
            "prompt": "a cozy mountain cabin at sunset, pine forest, warm light in windows, detailed, high quality",
            "size": "768x768",
            "n": 1,
            "quality": "medium",
        }, timeout=600)

        health = http_get("http://127.0.0.1:8100/health")
        print(f"Health after loading: {health}")
        if health.get("model") != "DreamShaper_8":
            print(f"FAIL: Expected model DreamShaper_8, got {health.get('model')}")
            return

        vram_after = nvidia_mem()
        print(f"VRAM after: {vram_after}")
        print("PASS: DreamShaper loaded successfully")

        # Test 4: Lazy load RealisticVision (switch from DreamShaper)
        print("\n=== Test 4: Lazy load RealisticVision (switch from DreamShaper) ===")
        vram_before = nvidia_mem()
        print(f"VRAM before switch: {vram_before}")

        gen = http_json("http://127.0.0.1:8100/v1/images/generations", {
            "model": "RealisticVision_V6_B1",
            "prompt": "portrait of a woman in soft natural window light, sharp focus, skin texture, photorealistic, high quality",
            "size": "768x768",
            "n": 1,
            "quality": "medium",
        }, timeout=600)

        health = http_get("http://127.0.0.1:8100/health")
        print(f"Health after switch: {health}")
        if health.get("model") != "RealisticVision_V6_B1":
            print(f"FAIL: Expected model RealisticVision_V6_B1, got {health.get('model')}")
            return

        vram_after = nvidia_mem()
        print(f"VRAM after switch: {vram_after}")
        print("PASS: Model switching successful")

        # Test 5: Lazy load MeinaMix (switch from RealisticVision)
        print("\n=== Test 5: Lazy load MeinaMix (switch from RealisticVision) ===")
        vram_before = nvidia_mem()
        print(f"VRAM before switch: {vram_before}")

        gen = http_json("http://127.0.0.1:8100/v1/images/generations", {
            "model": "MeinaMix_V11",
            "prompt": "anime girl with long blue hair standing in a cherry blossom garden, detailed eyes, vibrant, high quality",
            "size": "512x768",
            "n": 1,
            "quality": "medium",
        }, timeout=600)

        health = http_get("http://127.0.0.1:8100/health")
        print(f"Health after switch: {health}")
        if health.get("model") != "MeinaMix_V11":
            print(f"FAIL: Expected model MeinaMix_V11, got {health.get('model')}")
            return

        vram_after = nvidia_mem()
        print(f"VRAM after switch: {vram_after}")
        print("PASS: Final model loaded successfully")

        # Test 6: Verify inpaint works after model switching
        print("\n=== Test 6: Inpaint after model switching ===")
        import base64, io
        from PIL import Image, ImageDraw

        # Create a simple test image
        img = Image.new('RGB', (512, 512), (128, 128, 128))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 100, 412, 412], fill=(255, 255, 255))

        # Save to bytes
        buf_i, buf_m = io.BytesIO(), io.BytesIO()
        img.save(buf_i, format="PNG")
        mask = Image.new("L", (512, 512), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.rectangle([150, 150, 402, 402], fill=255)
        mask.save(buf_m, format="PNG")

        inp = http_json("http://127.0.0.1:8100/v1/images/inpaint", {
            "model": "MeinaMix_V11",
            "image": base64.b64encode(buf_i.getvalue()).decode(),
            "mask": base64.b64encode(buf_m.getvalue()).decode(),
            "prompt": "a beautiful garden fountain, detailed",
            "width": 512,
            "height": 512,
        }, timeout=600)

        if "image" not in inp:
            print(f"FAIL: Inpaint failed: {inp}")
            return
        print("PASS: Inpaint works after model switching")

        # Test 7: Verify /v1/models still lists all models after loading
        print("\n=== Test 7: /v1/models after loading models ===")
        models = http_get("http://127.0.0.1:8100/v1/models")
        print(f"Models: {json.dumps(models, indent=2)}")
        if len(models.get("data", [])) != 3:
            print(f"FAIL: Expected 3 models, got {len(models.get('data', []))}")
            return

        print("\n=== ALL TESTS PASSED ===")
        print("Lazy loading implementation is working correctly!")

    finally:
        # Stop the server
        print("\n=== Stopping server ===")
        run_powershell([".\Stop-Diffusion.ps1"])

if __name__ == "__main__":
    main()
