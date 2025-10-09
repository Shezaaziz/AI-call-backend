import time
import numpy as np
import io
import librosa
import soundfile as sf # Used for safe audio I/O on deployment platforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="AI Vishing Detection Backend",
    description="Provides real-time reputation analysis for numbers and voice analysis for audio files."
)

# --- Schemas for Number Analysis (Layer 1: Predictive Defense) ---

class NumberAnalysisRequest(BaseModel):
    phone_number: str

class AnalysisResult(BaseModel):
    classification: str
    confidence: float
    message: str

# --- Core Detection Logic ---

def run_reputation_analysis(number: str) -> AnalysisResult:
    """
    Simulates a machine learning model analyzing a phone number's reputation
    based on calling patterns (metadata).
    """
    # Simulate network/AI processing time (CRITICAL for real-time testing)
    time.sleep(0.5)

    # Simplified Decision Logic (MOCK ML MODEL)
    if number.endswith("9999"):
        return AnalysisResult(
            classification="AI_SCAM", 
            confidence=0.98, 
            message="High Confidence: Synthetic Voice/Vishing Pattern Detected."
        )
    elif number.endswith("5555"):
        return AnalysisResult(
            classification="SPAM",
            confidence=0.07,
            message="Low Confidence: Known Telemarketer/Robocall Number."
        )
    else:
        return AnalysisResult(
            classification="NORMAL", 
            confidence=0.0, 
            message="Number passed reputation check."
        )

# --- Endpoints ---

@app.get("/")
def read_root():
    """Health check endpoint to confirm the server is running."""
    return {"status": "Service Running", "message": "Access /docs for the API documentation."}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_number(request: NumberAnalysisRequest):
    """
    Analyzes the phone number metadata for pre-answer detection (Layer 1).
    """
    return run_reputation_analysis(request.phone_number)

@app.post("/analyze_voice", response_model=AnalysisResult)
async def analyze_voice(audio_file: UploadFile = File(...)):
    """
    Analyzes an audio snippet (MP4/AAC) for deepfake characteristics (Layer 2)
    using real acoustic feature extraction (MFCCs).
    """
    # Simulate AI processing time
    time.sleep(1.5) 
    
    try:
        # Read the file content
        contents = await audio_file.read()
        
        # 1. Load audio data using soundfile (safer I/O)
        audio_buffer = io.BytesIO(contents)
        audio_data, sr = sf.read(audio_buffer)

        # 2. Resample and ensure mono channel (standard ML preprocessing)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1) # Convert to mono

        # Librosa feature extraction requires 16000 Hz often, resample if necessary
        if sr != 16000:
             audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
             sr = 16000
        
        # 3. Feature Extraction (Mel-Frequency Cepstral Coefficients - Acoustic Fingerprint)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        
        # 4. Acoustic Analysis (Measure Stability/Smoothness - a common deepfake artifact)
        # Calculate the variance of the mean MFCCs. Synthesized voices often show less variance.
        mfcc_variance = np.var(np.mean(mfccs, axis=1))

        # 5. Decision Based on Feature (MOCK DECISION LOGIC based on MFCC Variance)
        
        # Threshold: Synthesized audio is often "too clean" (low variance).
        SYNTHETIC_THRESHOLD = 0.001 
        
        if mfcc_variance < SYNTHETIC_THRESHOLD:
            # Low variance suggests synthetic smoothness
            return AnalysisResult(
                classification="VOICE_DEEPFAKE",
                confidence=0.95,
                message=f"Deepfake Risk: Acoustic variance is low ({mfcc_variance:.6f}), suggesting synthetic generation."
            )
        else:
            # Higher variance suggests human speech irregularities
            return AnalysisResult(
                classification="HUMAN",
                confidence=0.99,
                message=f"Voice verified as human (Variance: {mfcc_variance:.6f})."
            )

    except Exception as e:
        print(f"ML Processing Error: {e}")
        # Return a structure that Kotlin can safely parse
        return AnalysisResult(
            classification="API_ERROR", 
            confidence=0.0, 
            message=f"ML Analysis Failed: {str(e)}"
        )