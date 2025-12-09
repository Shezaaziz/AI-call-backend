import time
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI(title="NoCapCalls AI Backend (Enhanced)", description="Memory-safe voice analysis with DSP features.")

# --- Schemas ---

class AnalysisResult(BaseModel):
    classification: str
    confidence: float
    message: str

# --- Enhanced Voice Analysis ---

def run_ml_inference(contents: bytes) -> AnalysisResult:
    """
    Enhanced DSP-based deepfake detection:
    Uses multiple acoustic features:
    - Standard deviation (signal variance)
    - Zero-crossing rate (ZCR)
    - Spectral flatness
    - Harmonic-to-noise ratio
    """
    try:
        audio = np.frombuffer(contents, dtype=np.int16).astype(np.float32)
        if audio.size == 0:
            return AnalysisResult("API_ERROR", 0.0, "Audio file empty.")

        # Normalize audio
        audio /= np.max(np.abs(audio)) + 1e-9

        # --- Feature 1: Standard deviation ---
        std_dev = np.std(audio)

        # --- Feature 2: Zero-Crossing Rate ---
        zcr = ((audio[:-1] * audio[1:]) < 0).sum() / len(audio)

        # --- Feature 3: Spectral Flatness ---
        fft = np.fft.rfft(audio)
        mag = np.abs(fft) + 1e-9
        spectral_flatness = (np.exp(np.mean(np.log(mag))) / np.mean(mag))

        # --- Feature 4: Harmonic-to-Noise Ratio approximation ---
        # Use auto-correlation peak
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        hnr = np.max(autocorr[1:]) / (np.mean(autocorr[1:]) + 1e-9)

        # --- Decision Logic ---
        score = 0
        # Low variance → synthetic
        if std_dev < 0.05: score += 1
        # Very low or very high ZCR → synthetic
        if zcr < 0.01 or zcr > 0.15: score += 1
        # Flat spectrum → synthetic
        if spectral_flatness > 0.5: score += 1
        # HNR too smooth → synthetic
        if hnr > 10: score += 1

        # Combine score into classification
        if score >= 2:
            classification = "VOICE_DEEPFAKE"
            confidence = min(0.95, 0.7 + 0.05 * score)
            message = f"Deepfake Risk Detected (Score: {score}, StdDev: {std_dev:.3f}, ZCR: {zcr:.3f}, Flatness: {spectral_flatness:.3f}, HNR: {hnr:.2f})"
        else:
            classification = "HUMAN"
            confidence = min(0.95, 0.7 + 0.05 * (4 - score))
            message = f"Voice verified as human (Score: {score}, StdDev: {std_dev:.3f}, ZCR: {zcr:.3f}, Flatness: {spectral_flatness:.3f}, HNR: {hnr:.2f})"

        return AnalysisResult(classification, confidence, message)

    except Exception as e:
        return AnalysisResult("API_ERROR", 0.0, f"ML Analysis Failed: {str(e)}")


# --- Endpoints ---

@app.post("/analyze_voice", response_model=AnalysisResult)
async def analyze_voice_endpoint(audio_file: UploadFile = File(...)):
    contents = await audio_file.read()
    return run_ml_inference(contents)


@app.get("/")
def read_root():
    return {"status": "Service Running", "message": "Access /docs for API documentation."}
