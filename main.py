import time
import numpy as np
import io
import math
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="NoCapCalls AI Backend (Stable)",
    description="Provides real-time reputation and memory-safe voice analysis."
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
    Simulates a machine learning model analyzing a phone number's reputation
    based on calling patterns (metadata).
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

def run_ml_inference(contents: bytes) -> AnalysisResult:
    """
    Performs memory-safe acoustic analysis using byte processing (bypassing librosa/torch memory limits).
    """
    time.sleep(1.0) # Simulate AI processing time
    
    try:
        # 1. Convert bytes to a numerical array (signed 16-bit integers are common for audio)
        audio_array = np.frombuffer(contents, dtype=np.int16)
        
        # 2. Calculate Segmentation Variance (Standard deviation of raw amplitude)
        if audio_array.size == 0:
             return AnalysisResult(
                classification="API_ERROR", 
                confidence=0.0, 
                message="Audio file was empty."
            )
             
        # Calculate standard deviation (measure of signal fluctuation/acoustic smoothness)
        std_dev = np.std(audio_array)

        # CRITICAL FIX: Extreme sensitivity value (0.1) for low-noise detection
        SYNTHETIC_THRESHOLD_LEAN = 0.1 
        
        if std_dev < SYNTHETIC_THRESHOLD_LEAN:
            # Low variance suggests synthetic smoothness
            return AnalysisResult(
                classification="VOICE_DEEPFAKE",
                confidence=0.92,
                message=f"Deepfake Risk: Signal smoothness detected (StdDev: {std_dev:.2f})."
            )
        else:
            return AnalysisResult(
                classification="HUMAN",
                confidence=0.90,
                message=f"Voice verified as human (StdDev: {std_dev:.2f})."
            )

    except Exception as e:
        print(f"ML Processing Error: {e}")
        # Return a structure that Kotlin can safely parse
        return AnalysisResult(
            classification="API_ERROR", 
            confidence=0.0, 
            message=f"ML Analysis Failed: {str(e)}"
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
    # Read file content safely as raw bytes
    contents = await audio_file.read()
    
    # Run the memory-safe inference function
    return run_ml_inference(contents)
