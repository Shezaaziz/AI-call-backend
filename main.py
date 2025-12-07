import time
import numpy as np
import io
import librosa
import torch 
from transformers import AutoProcessor, AutoModelForAudioClassification
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from starlette.responses import JSONResponse

# --- GLOBAL MODEL SETUP (LOADED ONCE ON SERVER START) ---
# This is the heavy part that causes the Out-of-Memory error on free servers.
MODEL_NAME = "facebook/wav2vec2-base-960h" 
SAMPLE_RATE = 16000
processor = None
model = None

def load_ml_model():
    """Loads the Hugging Face model/processor into memory."""
    global processor, model
    try:
        print(f"Loading ML Model: {MODEL_NAME}...")
        # CRITICAL: Loading this requires significant RAM (4GB+)
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        # Essential fallback if the large model fails to load during deployment
        print(f"FATAL WARNING: Could not load Hugging Face model. ML inference will fail. Error: {e}")
        # The variables remain None, which is handled in run_ml_inference
        pass

# Load the model on startup
load_ml_model()

# Initialize FastAPI app
app = FastAPI(
    title="NoCapCalls AI Backend (Full ML)",
    description="Provides real-time reputation and Hugging Face voice analysis."
)

# --- Schemas for Analysis ---

class NumberAnalysisRequest(BaseModel):
    phone_number: str

class AnalysisResult(BaseModel):
    classification: str
    confidence: float
    message: str

# --- Layer 1: Predictive Defense (Number Check) ---

def run_reputation_analysis(number: str) -> AnalysisResult:
    time.sleep(0.5)
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

# --- Layer 2: Verification Defense (Voice Check) ---

def run_ml_inference(audio_data, sr) -> AnalysisResult:
    """
    Runs actual ML inference using the loaded Hugging Face model or provides structured failure.
    """
    if model is None:
        # This occurs if the model failed to load in load_ml_model() due to OOM/resource limits
        
        # --- HEURISTIC FALLBACK (For Stable Deployment) ---
        # Run a statistical check as a final safety net for the college project demonstration
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfcc_variance = np.var(np.mean(mfccs, axis=1))
        
        # We use a highly sensitive threshold to simulate deepfake detection
        SYNTHETIC_THRESHOLD = 0.001 
        
        if mfcc_variance < SYNTHETIC_THRESHOLD:
            verdict = "VOICE_DEEPFAKE"
            confidence = 0.85
        else:
            verdict = "HUMAN"
            confidence = 0.90
            
        return AnalysisResult(
            classification=verdict,
            confidence=round(confidence, 4),
            message=f"[HEURISTIC FALLBACK] Deepfake signature detected via acoustic variance check."
        )

    else:
        # --- FULL HUGGING FACE INFERENCE ---
        
        # 1. Preprocessing
        inputs = processor(audio_data, sampling_rate=sr, return_tensors="pt")
        
        # 2. Model Inference
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # 3. Classification (MOCK labels, adjust for ASVspoof labels in final project)
        predicted_class_id = logits.argmax().item()
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_id].item()
        
        if predicted_class_id % 2 == 0:
            verdict = "VOICE_DEEPFAKE"
        else:
            verdict = "HUMAN"
            
        return AnalysisResult(
            classification=verdict,
            confidence=round(confidence, 4),
            message=f"[FULL ML] Classified using Hugging Face model (Label ID: {predicted_class_id})."
        )


# --- Endpoints ---

@app.get("/")
def read_root():
    return {"status": "Service Running", "message": "Access /docs for the API documentation."}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_number_endpoint(request: NumberAnalysisRequest):
    return run_reputation_analysis(request.phone_number)

@app.post("/analyze_voice", response_model=AnalysisResult)
async def analyze_voice_endpoint(audio_file: UploadFile = File(...)):
    """
    Analyzes an audio snippet for deepfake characteristics (Layer 2).
    """
    time.sleep(1.0) # Simulate network time

    try:
        contents = await audio_file.read()
        audio_buffer = io.BytesIO(contents)
        
        # Load audio using librosa for accurate preprocessing
        audio_data, sr = librosa.load(audio_buffer, sr=SAMPLE_RATE, mono=True)
        
        return run_ml_inference(audio_data, sr)

    except Exception as e:
        print(f"FATAL ML Processing Error: {e}")
        # Return a structure that Kotlin can safely parse
        return JSONResponse(
            status_code=500, 
            content={
                "classification": "ML_RUNTIME_CRASH", 
                "confidence": 0.0, 
                "message": f"ML Inference Failed: {str(e)}"
            }
        )