import time
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import torch
import torchaudio

# --- FastAPI app ---
app = FastAPI(
    title="NoCapCalls AI Backend (Hybrid)",
    description="Real-time number + voice analysis with hybrid Layer 2 detection."
)

# --- Schemas ---
class NumberAnalysisRequest(BaseModel):
    phone_number: str

class AnalysisResult(BaseModel):
    classification: str
    confidence: float
    message: str

# --- Layer 1: Number reputation check ---
def run_reputation_analysis(number: str) -> AnalysisResult:
    time.sleep(0.5)  # simulate latency
    if number.endswith("9999"):
        return AnalysisResult(
            classification="AI_SCAM", 
            confidence=0.98, 
            message="High Confidence: Synthetic Voice/Vishing Pattern Detected."
        )
    else:
        return AnalysisResult(
            classification="NORMAL", 
            confidence=0.0, 
            message="Number passed reputation check."
        )

# --- Load Tiny ML Model for Layer 2 (optional) ---
try:
    ml_model = torch.jit.load("tiny_deepfake_model.pt")
    ml_model.eval()
    print("[INFO] Tiny ML model loaded successfully.")
except Exception as e:
    ml_model = None
    print(f"[WARNING] Could not load ML model: {e}")

# --- Heuristic Pre-filter ---
def is_suspicious_audio(contents: bytes) -> bool:
    audio_array = np.frombuffer(contents, dtype=np.int16)
    if audio_array.size == 0:
        return False
    std_dev = np.std(audio_array)
    SYNTHETIC_THRESHOLD = 0.15
    return std_dev < SYNTHETIC_THRESHOLD

# --- Tiny ML Inference ---
def run_tiny_model(contents: bytes) -> AnalysisResult:
    if ml_model is None:
        return AnalysisResult(
            classification="VOICE_DEEPFAKE",
            confidence=0.5,
            message="ML model unavailable. Defaulting to Deepfake."
        )
    audio_tensor, sr = torchaudio.load(io.BytesIO(contents))
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_tensor = resampler(audio_tensor)
    with torch.no_grad():
        logits = ml_model(audio_tensor)
        prob = torch.sigmoid(logits).item() if isinstance(logits, torch.Tensor) else float(logits)
        if prob > 0.5:
            return AnalysisResult(
                classification="VOICE_DEEPFAKE",
                confidence=prob,
                message="Deepfake detected by ML model."
            )
        else:
            return AnalysisResult(
                classification="HUMAN",
                confidence=1 - prob,
                message="Verified human voice by ML model."
            )

# --- Layer 2: Hybrid Voice Check ---
def run_ml_inference(contents: bytes) -> AnalysisResult:
    if not is_suspicious_audio(contents):
        return AnalysisResult(
            classification="HUMAN",
            confidence=0.90,
            message="Heuristic check passed (likely human)."
        )
    return run_tiny_model(contents)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Service Running", "message": "Access /docs for the API documentation."}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_number_endpoint(request: NumberAnalysisRequest):
    return run_reputation_analysis(request.phone_number)

@app.post("/analyze_voice", response_model=AnalysisResult)
async def analyze_voice_endpoint(audio_file: UploadFile = File(...)):
    contents = await audio_file.read()
    return run_ml_inference(contents)
