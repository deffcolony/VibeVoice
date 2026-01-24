"""
VibeVoice API Server
Universal Compatibility: AllTalk (SillyTavern Native) + OpenAI Standard
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
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# VibeVoice Imports
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

app = FastAPI(title="VibeVoice API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Global State ---
model = None
processor = None
voice_presets = {}
args = None
IS_STREAMING_MODEL = False

class VoiceMapper:
    def __init__(self, voices_dir, is_streaming):
        self.presets = {}
        if not os.path.exists(voices_dir):
            alt_dir = os.path.join(os.getcwd(), "demo", "voices")
            if is_streaming: alt_dir = os.path.join(alt_dir, "streaming_model")
            if os.path.exists(alt_dir): voices_dir = alt_dir

        exts = ('.pt',) if is_streaming else ('.wav', '.mp3', '.flac', '.ogg')
        
        if os.path.exists(voices_dir):
            files = [f for f in os.listdir(voices_dir) if f.lower().endswith(exts)]
            print(f"[API] Found {len(files)} voices in {voices_dir}")
            for f in files:
                full_path = os.path.join(voices_dir, f)
                self.presets[f] = full_path 
                clean_name = os.path.splitext(f)[0] 
                self.presets[clean_name] = full_path
                simple_name = clean_name
                if '_' in simple_name: simple_name = simple_name.split('_')[0]
                if '-' in simple_name: simple_name = simple_name.split('-')[-1]
                self.presets[simple_name] = full_path

    def get_path(self, name):
        name = os.path.splitext(name)[0]
        if name in self.presets: return self.presets[name]
        name_lower = name.lower()
        for k, v in self.presets.items():
            if k.lower() == name_lower: return v
        if self.presets: return self.presets[list(self.presets.keys())[0]]
        return None

    def list_voices(self):
        keys = [os.path.basename(p) for p in self.presets.values()]
        return sorted(list(set(keys)))

# --- Inference Logic ---
def run_inference(text, speaker_name):
    formatted_text = f"Speaker 0: {text}"
    voice_path = voice_presets.get_path(speaker_name)
    if not voice_path:
        def_name = "Emma" if IS_STREAMING_MODEL else "Alice"
        voice_path = voice_presets.get_path(def_name)

    target_device = args.device
    if target_device == "cuda" and not torch.cuda.is_available(): target_device = "cpu"

    start_t = time.time()
    if IS_STREAMING_MODEL:
        cached_prompt = torch.load(voice_path, map_location=target_device, weights_only=False)
        inputs = processor.process_input_with_cached_prompt(
            text=formatted_text, cached_prompt=cached_prompt, padding=True,
            return_tensors="pt", return_attention_mask=True
        )
        for k, v in inputs.items():
            if torch.is_tensor(v): inputs[k] = v.to(target_device)
            
        outputs = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=args.cfg,
            tokenizer=processor.tokenizer, generation_config={'do_sample': False},
            all_prefilled_outputs=copy.deepcopy(cached_prompt)
        )
    else:
        inputs = processor(
            text=[formatted_text], voice_samples=[[voice_path]],
            padding=True, return_tensors="pt", return_attention_mask=True
        )
        for k, v in inputs.items():
            if torch.is_tensor(v): inputs[k] = v.to(target_device)
            
        outputs = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=args.cfg,
            tokenizer=processor.tokenizer, generation_config={'do_sample': False},
            is_prefill=True 
        )
    
    # print(f"[GEN] Time: {time.time()-start_t:.2f}s")

    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio_tensor = outputs.speech_outputs[0]
        if torch.is_tensor(audio_tensor):
            if audio_tensor.dtype == torch.bfloat16: audio_tensor = audio_tensor.float()
            audio_data = audio_tensor.detach().cpu().numpy()
        else:
            audio_data = audio_tensor
            
        if len(audio_data.shape) > 1: audio_data = audio_data.squeeze()
        max_val = np.abs(audio_data).max()
        if max_val > 1.0: audio_data = audio_data / max_val
        return audio_data
    return None

# --- OPENAI COMPATIBILITY ENDPOINTS ---

class OpenAIRequest(BaseModel):
    model: str = "vibevoice"
    input: str
    voice: str = "Emma"
    response_format: str = "wav"
    speed: float = 1.0 

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "vibevoice", "object": "model", "owned_by": "microsoft"}]}

@app.get("/v1/audio/voices")
def get_openai_voices():
    return {"voices": voice_presets.list_voices()}

@app.post("/v1/audio/speech")
async def openai_speech(req: OpenAIRequest):
    try:
        audio_data = run_inference(req.input, req.voice)
        if audio_data is None: raise HTTPException(500, "No audio generated")
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 24000, format='wav')
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        print(f"[OpenAI Error] {e}")
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
def get_alltalk_voices(): return {"voices": voice_presets.list_voices()}

@app.get("/api/rvcvoices")
def get_rvc_voices(): return {"rvcvoices": []}

@app.post("/api/previewvoice/")
async def preview_voice(voice: str = Form(...)):
    try:
        audio_data = run_inference("This is a preview of the voice.", voice)
        if audio_data is None: return JSONResponse({"status": "error"}, 500)
        
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"preview_{int(time.time())}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio_data, 24000)
        
        file_url = f"/outputs/{filename}" 
        return JSONResponse({
            "status": "generate-success", "output_file_path": filepath, "output_file_url": file_url
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, 500)

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
        if audio_data is None: return JSONResponse({"status": "error"}, 500)
        
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{int(time.time())}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio_data, 24000)
        
        file_url = f"/outputs/{filename}"
        return JSONResponse({
            "status": "generate-success", "output_file_path": filepath, "output_file_url": file_url
        })
    except Exception as e:
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
    if device == "cuda" and not torch.cuda.is_available(): device = "cpu"
    print(f"Loading {args.model_path} on {device}...")

    if "streaming" in args.model_path.lower() or "0.5b" in args.model_path.lower():
        IS_STREAMING_MODEL = True
        voices_path = os.path.join(os.getcwd(), "demo", "voices", "streaming_model")
        processor = VibeVoiceStreamingProcessor.from_pretrained(args.model_path)
        ModelClass = VibeVoiceStreamingForConditionalGenerationInference
    else:
        IS_STREAMING_MODEL = False
        voices_path = os.path.join(os.getcwd(), "demo", "voices")
        processor = VibeVoiceProcessor.from_pretrained(args.model_path)
        ModelClass = VibeVoiceForConditionalGenerationInference

    voice_presets = VoiceMapper(voices_path, IS_STREAMING_MODEL)
    
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attn = "flash_attention_2" if device == "cuda" else "sdpa"
    
    try:
        model = ModelClass.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device if device!="mps" else None, attn_implementation=attn)
    except:
        model = ModelClass.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device if device!="mps" else None, attn_implementation="sdpa")
    
    if device == "mps": model.to("mps")
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.steps)
    print(">>> VibeVoice API Ready (Universal Mode) <<<")
    uvicorn.run(app, host=args.host, port=args.port)