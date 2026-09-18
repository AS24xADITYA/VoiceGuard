import json
import time
from pathlib import Path
import httpx

import os
import io
import wave
import math
import struct

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")

def get_or_create_test_audio(tmp_path: Path = Path("test_sample.wav")) -> Path:
    """Find an existing audio sample or generate a synthetic 16kHz WAV clip."""
    # Check for existing canonical or test audio in artifacts
    for p in Path("data").glob("**/*.wav"):
        if p.exists() and p.stat().st_size > 1000:
            return p
    # Otherwise generate a 4-second 16kHz sine wave clip
    if not tmp_path.exists():
        sr = 16000
        duration = 4.0
        n_samples = int(sr * duration)
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            for i in range(n_samples):
                val = int(math.sin(2 * math.pi * 440.0 * i / sr) * 16000)
                wf.writeframes(struct.pack("<h", val))
    return tmp_path

def run():
    audio_path = get_or_create_test_audio()
    print(f"Testing with audio file: {audio_path} (exists={audio_path.exists()}, size={audio_path.stat().st_size} bytes)")
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        res = httpx.post(f"{BASE_URL}/analyses", files=files, timeout=30.0)
    
    print(f"Upload response: {res.status_code}")
    data = res.json()
    print("Initial response:", json.dumps(data, indent=2))
    analysis_id = data["id"]
    
    # Poll until complete
    start_poll = time.time()
    while time.time() - start_poll < 60:
        poll_res = httpx.get(f"{BASE_URL}/analyses/{analysis_id}", timeout=10.0)
        poll_data = poll_res.json()
        status = poll_data.get("status")
        stage = poll_data.get("stage")
        pct = poll_data.get("progress_pct")
        print(f"Polling: status={status}, stage={stage}, pct={pct}")
        if status in ("COMPLETE", "FAILED"):
            break
        time.sleep(1.5)
        
    full_res = httpx.get(f"{BASE_URL}/analyses/{analysis_id}", timeout=10.0)
    full_data = full_res.json()
    print("\n================ FULL COMPLETED ANALYSIS JSON ================")
    print(json.dumps(full_data, indent=2))
    print("==============================================================\n")
    
    # Check artifacts
    artifacts = full_data.get("artifacts", [])
    print(f"Artifacts count: {len(artifacts)}")
    for a in artifacts:
        print(f"  Artifact ID={a.get('id')}, kind={a.get('kind')}, url={a.get('download_url')}")
        if a.get('download_url'):
            art_url = a['download_url']
            if art_url.startswith("/"):
                art_url = f"http://127.0.0.1:8000{art_url}"
            try:
                r_art = httpx.get(art_url, timeout=5.0)
                print(f"    Fetch status: {r_art.status_code}, size: {len(r_art.content)} bytes")
            except Exception as e:
                print(f"    Fetch failed: {e}")

    # Challenge-Response round
    print("\n--- Testing Challenge-Response Round ---")
    chal_issue_res = httpx.post(f"{BASE_URL}/analyses/{analysis_id}/challenges", timeout=10.0)
    print(f"Challenge issue status: {chal_issue_res.status_code}")
    chal_data = chal_issue_res.json()
    print("Challenge issued:", json.dumps(chal_data, indent=2))
    chal_id = chal_data.get("id")
    
    if chal_id:
        with open(audio_path, "rb") as f_resp:
            resp_files = {"file": ("challenge_resp.wav", f_resp, "audio/wav")}
            verify_res = httpx.post(
                f"{BASE_URL}/challenges/{chal_id}/respond",
                files=resp_files,
                timeout=30.0
            )
            print(f"Challenge respond status: {verify_res.status_code}")
            verify_data = verify_res.json()
            print("Challenge verification result:", json.dumps(verify_data, indent=2))

if __name__ == "__main__":
    run()
