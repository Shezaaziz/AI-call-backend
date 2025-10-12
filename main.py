import time
import numpy as np
import io
import librosa
import soundfile as sf
import torch 
from transformers import AutoProcessor, AutoModelForAudioClassification, pipeline 
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# --- GLOBAL MODEL SETUP (LOADED ONCE ON SERVER START) ---
# NOTE: This model is a general-purpose audio classification model for demonstration.
try:
    MODEL_NAME = "facebook/wav2vec2-base-960h" 
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    print(f"--- ML Model '{MODEL_NAME}' Loaded Successfully ---")
except Exception as e:
    # Essential fallback if the large model fails to load during deployment
    print(f"WARNING: Could not load Hugging Face model. Falling back to Heuristic Mode. Error: {e}")
    model = None
    processor = None


# Initialize FastAPI app
app = FastAPI(
    title="AI Vishing Detection Backend",
    description="Provides real-time reputation analysis for numbers and voice analysis for audio files."
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
    """
    Simulates a machine learning model analyzing a phone number's reputation.
    """
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
    Runs actual ML inference using the loaded Hugging Face model or falls back to Heuristic.
    """
    if model is None:
        # --- HEURISTIC FALLBACK (Corrected for higher sensitivity) ---
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfcc_variance = np.var(np.mean(mfccs, axis=1))
        
        # CRITICAL FIX: Increased threshold to 0.1 for high sensitivity
        SYNTHETIC_THRESHOLD = 0.1 
        
        if mfcc_variance < SYNTHETIC_THRESHOLD:
            # Low variance suggests synthetic smoothness
            return AnalysisResult(
                classification="VOICE_DEEPFAKE",
                confidence=0.92,
                message=f"[HEURISTIC FALLBACK] Too smooth, suggests synthesis (Var: {mfcc_variance:.6f})."
            )
        else:
            return AnalysisResult(
                classification="HUMAN",
                confidence=0.90,
                message=f"[HEURISTIC FALLBACK] Human-like acoustic variance (Var: {mfcc_variance:.6f})."
            )

    else:
        # --- FULL HUGGING FACE INFERENCE ---
        
        # 1. Preprocessing and Feature Extraction
        inputs = processor(audio_data, sampling_rate=sr, return_tensors="pt")
        
        # 2. Model Inference
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # 3. Classification (MOCK: Update with specific ASVspoof labels in final project)
        predicted_class_id = logits.argmax().item()
        
        if predicted_class_id % 2 == 0:
            verdict = "VOICE_DEEPFAKE"
            confidence = 0.95
        else:
            verdict = "HUMAN"
            confidence = 0.99
            
        return AnalysisResult(
            classification=verdict,
            confidence=confidence,
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
        
        # Use soundfile/librosa for safe reading and resampling
        audio_data, sr = sf.read(audio_buffer)
        
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 16000 Hz, which is standard for Wav2Vec2 models
        if sr != 16000:
             audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
             sr = 16000
        
        return run_ml_inference(audio_data, sr)

    except Exception as e:
        print(f"ML Processing Error: {e}")
        # Return a structure that Kotlin can safely parse
        return AnalysisResult(
            classification="API_ERROR", 
            confidence=0.0, 
            message=f"ML Analysis Failed: {str(e)}"
        )
    