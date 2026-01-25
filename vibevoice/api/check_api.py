"""
VibeVoice Diagnostic Tool
Checks connection, voice availability, and audio generation latency.
"""
import requests
import json
import time
import os
import sys

# --- CONFIG ---
PORT = 7985
BASE_URL = f"http://127.0.0.1:{PORT}"
OUTPUT_DIR = "diagnostics"

# --- COLORS ---
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"

# Enable VT100 emulation on Windows for colors
os.system('')

def print_header(title):
    print(f"\n{CYAN}=========================================={RESET}")
    print(f"{BOLD} {title} {RESET}")
    print(f"{CYAN}=========================================={RESET}")

def print_pass(msg):
    print(f"[{GREEN}PASS{RESET}] {msg}")

def print_fail(msg, detail=None):
    print(f"[{RED}FAIL{RESET}] {msg}")
    if detail:
        print(f"       {YELLOW}Detail: {detail}{RESET}")

def ensure_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def main():
    print_header("VIBEVOICE DIAGNOSTIC TOOL")
    print(f"Target URL: {YELLOW}{BASE_URL}{RESET}\n")
    
    ensure_dir()
    
    # 1. Health Check
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            print_pass("Server is reachable")
        else:
            print_fail(f"Server returned {r.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print_fail("Could not connect to server.", "Is it running? Check Option 2 in Launcher.")
        return

    # 2. Get Voices (AllTalk Protocol)
    detected_voice = None
    try:
        r = requests.get(f"{BASE_URL}/api/voices")
        if r.status_code == 200:
            data = r.json()
            voices = data.get("voices", [])
            print_pass(f"AllTalk Protocol: Found {len(voices)} voices")
            if voices:
                detected_voice = voices[0]
                print(f"       First voice: {CYAN}{detected_voice}{RESET}")
        else:
            print_fail("AllTalk Protocol: Voices endpoint failed", r.text)
    except Exception as e:
        print_fail("AllTalk Protocol Error", str(e))

    # 3. Get Voices (OpenAI Protocol)
    try:
        r = requests.get(f"{BASE_URL}/v1/audio/voices")
        if r.status_code == 200:
            data = r.json()
            voices = data.get("voices", [])
            print_pass(f"OpenAI Protocol: Found {len(voices)} voices")
        else:
            print_fail("OpenAI Protocol: Voices endpoint failed", r.text)
    except Exception as e:
        print_fail("OpenAI Protocol Error", str(e))

    if not detected_voice:
        print_fail("No voices detected. Defaulting to 'Alice'.")
        detected_voice = "Alice" 

    # 4. Test Generation (AllTalk)
    print_header("TEST 1: ALLTALK GENERATION")
    payload = {
        "text_input": "This is a diagnostic test for the AllTalk protocol.",
        "character_voice_gen": detected_voice,
        "language": "en"
    }
    
    try:
        start = time.time()
        r = requests.post(f"{BASE_URL}/api/tts-generate", data=payload)
        latency = time.time() - start
        
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "generate-success":
                print_pass(f"Generated successfully in {latency:.2f}s")
                url = data.get("output_file_url")
                print(f"       Path: {data.get('output_file_path')}")
                print(f"       URL:  {url}")
            else:
                print_fail("Generation status returned failure", data)
        else:
            print_fail(f"HTTP Error {r.status_code}", r.text)
    except Exception as e:
        print_fail("AllTalk Generation Exception", str(e))

    # 5. Test Generation (OpenAI)
    print_header("TEST 2: OPENAI GENERATION")
    payload = {
        "input": "This is a diagnostic test for the OpenAI protocol.",
        "voice": detected_voice,
        "model": "vibevoice"
    }
    
    try:
        start = time.time()
        r = requests.post(f"{BASE_URL}/v1/audio/speech", json=payload)
        latency = time.time() - start
        
        if r.status_code == 200:
            size_kb = len(r.content) / 1024
            print_pass(f"Generated successfully in {latency:.2f}s")
            print(f"       Size: {size_kb:.1f} KB")
            
            filepath = os.path.join(OUTPUT_DIR, "openai_test.wav")
            with open(filepath, "wb") as f:
                f.write(r.content)
            print(f"       Saved to: {filepath}")
        else:
            print_fail(f"HTTP Error {r.status_code}", r.text)
    except Exception as e:
        print_fail("OpenAI Generation Exception", str(e))

    print("\n" + "="*30)
    print("DIAGNOSTIC COMPLETE")
    print("="*30)

if __name__ == "__main__":
    main()