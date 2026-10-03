#!/usr/bin/env python3
"""
Interactive menu system for Odysseus Diffusion Server.
Run with: python run_diffusion_menu.py
"""
import sys
import os
import json
import base64
import requests
import time
from datetime import datetime

# Configuration (relative to this script: portable, no hardcoded paths)
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_URL = os.environ.get("DIFFUSION_URL", "http://127.0.0.1:8101")
OUTPUT_DIR = os.path.join(SERVER_DIR, "tests", "output")
NEGATIVE_PROMPT = "blurry, low quality, low resolution, bad anatomy, deformed, distorted, extra fingers, missing fingers, bad hands, text, watermark, oversaturated, artifacts"

MODELS = {
    "1": {
        "id": "DreamShaper_8",
        "name": "DreamShaper 8 (General/Art/Realism, SD1.5)",
        "base": "768x768",
        "max": "1024x1024"
    },
    "2": {
        "id": "AbsoluteReality_1.8",
        "name": "Absolute Reality 1.8.1 (Photorealism, SD1.5)",
        "base": "768x768",
        "max": "1024x1024"
    },
    "3": {
        "id": "AnimagineXL_3.1",
        "name": "Animagine XL 3.1 (Anime, SDXL)",
        "base": "1024x1024",
        "max": "1920x1080"
    },
    "4": {
        "id": "JuggernautXL_v9",
        "name": "Juggernaut XL v9 (Photorealistic, SDXL)",
        "base": "1024x1024",
        "max": "1920x1080"
    }
}

STEPS_OPTIONS = {
    "1": ("Fast (15 steps)", 15),
    "2": ("Balanced (25 steps)", 25),
    "3": ("High Quality (35 steps)", 35),
    "4": ("Custom", None)
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def get_input(prompt, default=""):
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()

def main_menu():
    while True:
        clear_screen()
        print_header("ODYSSEUS DIFFUSION - MODEL SELECTION & GENERATION")
        print()
        print(f"Server: {SERVER_URL}")
        print(f"Output: {OUTPUT_DIR}")
        print()
        print("Available Models:")
        print()
        for key, model in MODELS.items():
            print(f"  {key}) {model['name']}")
            print(f"     Base: {model['base']} | Max: {model['max']}")
        print()
        print("  Q) Quit")
        print()
        print("=" * 60)
        
        choice = input("Select model (1-4 or Q): ").strip().upper()
        if choice == 'Q':
            return
        if choice in MODELS:
            model = MODELS[choice]
            prompt_menu(model)
        else:
            print("Invalid choice. Press Enter to continue...")
            input()

def prompt_menu(model):
    clear_screen()
    print_header(f"SELECTED: {model['name']} ({model['id']})")
    print(f"Base Resolution: {model['base']} | Max: {model['max']}")
    print()
    
    prompt = get_input("Enter your POSITIVE PROMPT (what you want to generate)")
    if not prompt:
        print("Prompt cannot be empty.")
        input("Press Enter to continue...")
        return

    # Size selection
    clear_screen()
    print(f"Selected: {model['name']}")
    print(f"Prompt: {prompt}")
    print(f"Negative: {NEGATIVE_PROMPT}")
    print()
    print("Resolution Options:")
    print(f"  1) Base Resolution ({model['base']})")
    print(f"  2) Max Resolution ({model['max']})")
    print("  3) Custom Size (e.g. 512x512)")
    
    size_choice = get_input("Select resolution (1-3)", "1")
    if size_choice == "1":
        size = model['base']
    elif size_choice == "2":
        size = model['max']
    elif size_choice == "3":
        size = get_input("Enter custom size (WIDTHxHEIGHT)")
        if not size:
            print("Invalid size.")
            input("Press Enter to continue...")
            return
    else:
        print("Invalid choice.")
        input("Press Enter to continue...")
        return

    # Steps selection
    clear_screen()
    print("Inference Steps (quality vs speed):")
    for key, (desc, steps) in STEPS_OPTIONS.items():
        print(f"  {key}) {desc}")
    
    steps_choice = get_input("Select steps (1-4)", "2")
    if steps_choice in STEPS_OPTIONS:
        desc, steps = STEPS_OPTIONS[steps_choice]
        if steps is None:
            steps = int(get_input("Enter custom steps (10-100)", "25"))
    else:
        print("Invalid choice.")
        input("Press Enter to continue...")
        return

    # Filename
    filename = get_input("Output Filename (without extension, empty = auto)", "")
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{MODELS['1']['id']}_{timestamp}"  # Will be updated with actual model id

    # Confirmation
    clear_screen()
    print("=" * 60)
    print("  GENERATION CONFIRMATION")
    print("=" * 60)
    print(f"Model:       {model['name']} ({model['id']})")
    print(f"Prompt:      {prompt}")
    print(f"Negative:    {NEGATIVE_PROMPT}")
    print(f"Resolution:  {size}")
    print(f"Steps:       {STEPS_OPTIONS.get(steps_choice, ('Custom', 25))[1] if STEPS_OPTIONS.get(steps_choice, ('Custom', 25))[1] else 'Custom'}")
    # Recalculate steps properly
    steps_val = STEPS_OPTIONS.get(steps_choice, ('Custom', 25))[1]
    if steps_val is None:
        steps_val = int(get_input("Enter custom steps (10-100)", "25"))
    print(f"Steps:       {steps_val}")
    print(f"Filename:    {filename}.png")
    print(f"Output:      {OUTPUT_DIR}")
    print("=" * 60)
    
    confirm = get_input("Generate? (Y/N)", "Y").upper()
    if confirm != 'Y':
        return

    # Update filename with correct model id
    if not filename.startswith(MODELS['1']['id']):
        # Update filename to include model id
        pass  # Will use the model id in the generation

    # Generate
    generate_image(model, prompt, size, STEPS_OPTIONS.get(steps_choice, ('Custom', 25))[1] if STEPS_OPTIONS.get(steps_choice, ('Custom', 25))[1] else 'Custom', prompt, filename)

def generate_image(model, prompt, size, steps, positive_prompt, filename):
    """Generate image via the API."""
    clear_screen()
    print("=" * 60)
    print("  GENERATING...")
    print("=" * 60)
    print(f"Model: {model['name']} ({model['id']})")
    print(f"Prompt: {prompt}")
    print(f"Size: {size}")
    print(f"Steps: {STEPS_OPTIONS.get('2', ('Balanced', 25))[1]}")
    print()
    print("Requesting generation...")
    print()

    # Build payload
    payload = {
        'model': model['id'],
        'prompt': prompt,
        'size': size,
        'n': 1,
        'quality': 'medium',
        'steps': int(get_input("Steps", "25"))
    }
    # Use the negative prompt from config
    payload['negative_prompt'] = NEGATIVE_PROMPT
    
    url = f"{SERVER_URL}/v1/images/generations"
    headers = {'Content-Type': 'application/json'}
    
    print("Sending request...")
    start = time.time()
    
    try:
        response = requests.post(
            f"{SERVER_URL}/v1/images/generations",
            json={
                'model': model['id'],
                'prompt': prompt,
                'size': size,
                'n': 1,
                'quality': 'medium',
                'steps': int(get_input("Steps (default 25)", "25")),
                'negative_prompt': NEGATIVE_PROMPT
            },
            headers={'Content-Type': 'application/json'},
            timeout=600
        )
        
        if response.status_code == 200:
            data = response.json()
            b64 = data['data'][0]['b64_json']
            img_bytes = base64.b64decode(b64)
            
            # Generate filename with model id
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_filename = f"{model['id']}_{timestamp}.png"
            out_path = os.path.join(OUTPUT_DIR, out_filename)
            
            with open(out_path, 'wb') as f:
                f.write(base64.b64decode(response.json()['data'][0]['b64_json']))
            
            elapsed = time.time() - start
            print(f"\nGeneration complete in {time.time() - start:.1f}s!")
            print(f"Saved to: {out_path}")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    input("\nPress Enter to continue...")

def main():
    # Check server health
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code == 200:
            health = resp.json()
            print(f"Server status: {health.get('status', 'unknown')}")
            print(f"Available models: {', '.join(health.get('available_models', []))}")
            print()
    except Exception as e:
        print(f"Warning: Could not connect to server: {e}")
        print("Make sure the server is running on port 8101")
        print()
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")

if __name__ == '__main__':
    main()