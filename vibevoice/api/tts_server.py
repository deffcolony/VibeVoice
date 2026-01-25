"""
VibeVoice API Server - Hybrid Edition
Combines: 
1. Official VibeVoice-API Robustness (Voice mapping, Audio pre-processing)
2. AllTalk Protocol Support (SillyTavern Native)
3. OpenAI Protocol Support
"""

import argparse
import os
import io
import torch
import uvicorn
import numpy as np
import soundfile as sf
import copy
import time
import json
import logging
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# VibeVoice Imports
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VibeVoiceAPI")

app = FastAPI(title="VibeVoice API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Global State ---
model = None
processor = None
voice_mapper = None
args = None
IS_STREAMING_MODEL = False

# --- UTILS: Audio & Voice Mapping (Ported from Official API) ---

def load_audio_to_numpy(file_path: str) -> np.ndarray:
    """Loads audio file to float32 numpy array, taking first channel if stereo."""
    try:
        data, sr = sf.read(file_path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0] # Take first channel
        return data.astype(np.float32)
    except Exception as e:
        logger.error(f"Failed to load audio {file_path}: {e}")
        raise

class RobustVoiceMapper:
    """
    Robust scanning of the voices directory. 
    Maps friendly names (Alice, en-Alice_woman) to absolute file paths.
    """
    def __init__(self, root_dirs: List[str]):
        self.map = {}
        self.root_dirs = root_dirs
        self.scan()

    def scan(self):
        self.map = {}
        audio_exts = {'.wav', '.mp3', '.flac', '.ogg'}
        
        for d in self.root_dirs:
            if not os.path.exists(d):
                continue
            
            logger.info(f"Scanning for voices in: {d}")
            for root, _, files in os.walk(d):
                for f in files:
                    if os.path.splitext(f)[1].lower() in audio_exts:
                        full_path = os.path.abspath(os.path.join(root, f))
                        file_name = os.path.splitext(f)[0]
                        
                        # Register exact filename
                        self.map[file_name] = full_path
                        
                        # Register simplified names (e.g. "Alice" from "en-Alice_woman")
                        parts = file_name.split('-')
                        if len(parts) > 1:
                            simple_name = parts[-1].split('_')[0] # Alice
                            self.map[simple_name] = full_path
                            self.map[parts[-1]] = full_path # Alice_woman
                        else:
                            self.map[file_name.split('_')[0]] = full_path

    def get_path(self, name: str) -> Optional[str]:
        if not name: return None
        # 1. Direct match
        if name in self.map: return self.map[name]
        # 2. Case insensitive
        name_lower = name.lower()
        for k, v in self.map.items():
            if k.lower() == name_lower: return v
        # 3. Partial match
        for k, v in self.map.items():
            if name_lower in k.lower(): return v
        # 4. Fallback: Return first available
        if self.map:
            first = list(self.map.values())[0]
            logger.warning(f"Voice '{name}' not found, falling back to {os.path.basename(first)}")
            return first
        return None

    def list_voices(self):
        # Return unique filenames sorted
        paths = set(self.map.values())
        names = [os.path.basename(p) for p in paths]
        return sorted(names)

# --- Inference Logic ---

def run_inference(text: str, speaker_name: str, speed: float = 1.0):
    """
    Main inference driver.
    Fixes the 'tuple index out of range' error by loading audio to numpy first.
    """
    voice_path = voice_mapper.get_path(speaker_name)
    if not voice_path:
        raise ValueError(f"No voices found. Ensure the 'demo/voices' folder exists.")

    logger.info(f"Generating for: '{text[:30]}...' using voice: {os.path.basename(voice_path)}")

    formatted_text = f"Speaker 0: {text}"
    target_device = args.device
    
    # 1. Load Audio Data (Crucial Fix: Processor needs Array, not String path)
    if IS_STREAMING_MODEL and voice_path.endswith('.pt'):
        # Special case for .pt latents in streaming model
        voice_input = torch.load(voice_path, map_location=target_device, weights_only=False)
    else:
        # Standard workflow: Load WAV to Numpy
        voice_input = load_audio_to_numpy(voice_path)

    start_t = time.time()

    # 2. Process Inputs
    if IS_STREAMING_MODEL:
        if isinstance(voice_input, torch.Tensor) or isinstance(voice_input, dict):
             # Cached prompt (.pt file)
            inputs = processor.process_input_with_cached_prompt(
                text=formatted_text, cached_prompt=voice_input, padding=True,
                return_tensors="pt", return_attention_mask=True
            )
            prefilled = copy.deepcopy(voice_input)
        else:
            # Raw Audio
            inputs = processor(
                text=[formatted_text], voice_samples=[[voice_input]],
                padding=True, return_tensors="pt", return_attention_mask=True
            )
            prefilled = None
    else:
        # Standard Model
        # NOTE: voice_samples expects a list of samples per batch item. 
        # Since batch size is 1, we pass [[numpy_array]].
        inputs = processor(
            text=[formatted_text], voice_samples=[[voice_input]],
            padding=True, return_tensors="pt", return_attention_mask=True
        )

    # Move inputs to device
    for k, v in inputs.items():
        if torch.is_tensor(v): inputs[k] = v.to(target_device)

    # 3. Generate
    with torch.inference_mode():
        generate_kwargs = {
            "max_new_tokens": None,
            "cfg_scale": args.cfg,
            "tokenizer": processor.tokenizer,
            "generation_config": {'do_sample': False},
        }

        if IS_STREAMING_MODEL:
             outputs = model.generate(**inputs, **generate_kwargs, all_prefilled_outputs=prefilled)
        else:
             outputs = model.generate(**inputs, **generate_kwargs, is_prefill=True)

    logger.info(f"Inference Time: {time.time()-start_t:.2f}s")

    # 4. Post-process Audio
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio_tensor = outputs.speech_outputs[0]
        if torch.is_tensor(audio_tensor):
            if audio_tensor.dtype == torch.bfloat16: audio_tensor = audio_tensor.float()
            audio_data = audio_tensor.detach().cpu().numpy()
        else:
            audio_data = audio_tensor
            
        if len(audio_data.shape) > 1: audio_data = audio_data.squeeze()
        
        # Apply normalization to prevent clipping
        max_val = np.abs(audio_data).max()
        if max_val > 1.0: audio_data = audio_data / max_val
        
        return audio_data
        
    return None

# --- OPENAI COMPATIBILITY ENDPOINTS ---

class OpenAIRequest(BaseModel):
    model: str = "vibevoice"
    input: str
    voice: str = "Alice"
    response_format: str = "wav"
    speed: float = 1.0 

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "vibevoice", "object": "model", "owned_by": "microsoft"}]}

@app.get("/v1/audio/voices")
def get_openai_voices():
    return {"voices": voice_mapper.list_voices()}

@app.post("/v1/audio/speech")
async def openai_speech(req: OpenAIRequest):
    try:
        audio_data = run_inference(req.input, req.voice, req.speed)
        if audio_data is None: raise HTTPException(500, "No audio generated")
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 24000, format='wav')
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        logger.error(f"OpenAI Handler Error: {e}")
        raise HTTPException(500, str(e))

# --- ALLTALK COMPATIBILITY ENDPOINTS ---

@app.get("/api/ready")
def api_ready(): return Response(content="Ready", media_type="text/plain")

@app.get("/api/currentsettings")
def api_settings():
    return {
        "models_available": [{"name": "VibeVoice", "model_name": "API TTS"}],
        "current_model_loaded": "VibeVoice",
        "deepspeed_available": False, "deepspeed_status": False,
        "low_vram_status": False, "finetuned_model": False,
        "engines_available": [], "current_engine_loaded": "VibeVoice",
        "deepspeed_capable": False, "deepspeed_enabled": False,
        "lowvram_capable": False, "lowvram_enabled": False
    }

@app.get("/api/voices")
def get_alltalk_voices(): return {"voices": voice_mapper.list_voices()}

@app.get("/api/rvcvoices")
def get_rvc_voices(): return {"rvcvoices": []}

@app.post("/api/previewvoice/")
async def preview_voice(voice: str = Form(...)):
    # Legacy endpoint mapping
    return await alltalk_speech(
        text_input="This is a preview of the selected voice.",
        character_voice_gen=voice
    )

@app.post("/api/tts-generate")
async def alltalk_speech(
    text_input: str = Form(...),
    character_voice_gen: str = Form(...),
    language: str = Form("en"),
    text_filtering: str = Form(None),
    output_file_name: str = Form(None),
):
    try:
        audio_data = run_inference(text_input, character_voice_gen)
        if audio_data is None: 
             return JSONResponse({"status": "error", "error": "Model returned no audio"}, 500)
        
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{int(time.time())}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Save to disk
        sf.write(filepath, audio_data, 24000)
        
        file_url = f"/outputs/{filename}"
        return JSONResponse({
            "status": "generate-success", "output_file_path": filepath, "output_file_url": file_url
        })
    except Exception as e:
        logger.exception("AllTalk Handler Error")
        return JSONResponse({"status": "generate-failure", "error": str(e)}, 500)

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/v1")
def catch_post_v1(): return {"status": "ok"}

# --- STARTUP ---
output_dir = os.path.join(os.getcwd(), "outputs")
os.makedirs(output_dir, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=output_dir), name="outputs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=1.5)
    args = parser.parse_args()

    # Model Loading
    device = args.device
    if device == "cuda" and not torch.cuda.is_available(): 
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
        
    print(f"Loading {args.model_path} on {device}...")

    # Determine voice directories
    root = os.getcwd()
    voice_dirs = [
        os.path.join(root, "demo", "voices"),
        os.path.join(root, "voices"),
        # Add relative path just in case we are deep in subfolders
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "demo", "voices"))
    ]

    if "streaming" in args.model_path.lower() or "0.5b" in args.model_path.lower():
        IS_STREAMING_MODEL = True
        logger.info("Mode: Streaming (Experimental)")
        # Prioritize streaming model latents
        voice_dirs.insert(0, os.path.join(root, "demo", "voices", "streaming_model"))
        processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_path)
        ModelClass = VibeVoiceStreamingForConditionalGenerationInference
    else:
        IS_STREAMING_MODEL = False
        logger.info("Mode: Standard")
        processor = VibeVoiceProcessor.from_pretrained(args.model_path)
        ModelClass = VibeVoiceForConditionalGenerationInference

    # Initialize Robust Voice Mapper
    voice_mapper = RobustVoiceMapper(voice_dirs)
    
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attn = "flash_attention_2" if device == "cuda" else "sdpa"
    
    try:
        model = ModelClass.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device if device!="mps" else None, attn_implementation=attn)
    except Exception as e:
        logger.warning(f"Failed to load with flash_attention_2 ({e}), falling back to sdpa")
        model = ModelClass.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device if device!="mps" else None, attn_implementation="sdpa")
    
    if device == "mps": model.to("mps")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.steps)
    
    print(">>> VibeVoice API Ready (Universal Mode) <<<")
    uvicorn.run(app, host=args.host, port=args.port)