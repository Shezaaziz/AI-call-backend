import io
import math
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Tuple
import soundfile as sf
import librosa

app = FastAPI(
    title="NoCapCalls AI Backend (Deterministic DSP Engine)",
    description="Memory-safe voice analysis using DSP features (no heavy ML)."
)

# --- Response schema ---
class AnalysisResult(BaseModel):
    classification: str
    confidence: float
    message: str

# ----------------------------
# Audio utilities / features
# ----------------------------

def safe_load_audio(contents: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Try to load audio bytes into a mono float32 numpy array at target_sr.
    Uses soundfile (pysoundfile) first, then falls back to librosa as needed.
    """
    try:
        # soundfile can read from BytesIO for many container formats (wav, flac, ogg, etc.)
        data, sr = sf.read(io.BytesIO(contents), dtype="float32")
        # sf.read may return 2-D for stereo; convert to mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        # ensure 1D float32
        data = data.astype(np.float32)
        return data, sr
    except Exception:
        # Last-resort: try librosa.load (librosa also uses soundfile internally if available)
        try:
            data, sr = librosa.load(io.BytesIO(contents), sr=target_sr, mono=True)
            data = data.astype(np.float32)
            return data, sr
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

def rmse_energy_jitter(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 512) -> float:
    """
    Compute frame energy jitter: standard deviation of frame RMS energies normalized by mean energy.
    Lower jitter often suggests synthetic smoothing; human speech typically has higher jitter.
    """
    # RMS per frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    mean_rms = np.mean(rms) + 1e-12
    jitter = np.std(rms) / mean_rms
    return float(jitter)

def spectral_entropy(audio: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> float:
    """
    Spectral entropy (Shannon) computed across averaged power spectrum frames.
    Higher entropy -> more "noisy/complex" spectrum (often human). Lower -> smoother likely synthetic.
    """
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))**2
    # average power spectral density across frames
    psd = np.mean(S, axis=1) + 1e-12
    psd_norm = psd / np.sum(psd)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    # normalize by maximum possible entropy (log2(#bins))
    max_ent = math.log2(len(psd_norm))
    return float(entropy / (max_ent + 1e-12))

def zero_crossing_rate(audio: np.ndarray, frame_length: int = 1024, hop_length: int = 512) -> float:
    z = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    return float(np.mean(z))

def pitch_variability(audio: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 500.0) -> float:
    """
    Use librosa.yin to estimate f0; compute coefficient of variation (std/mean) of voiced frames.
    Low variability (very flat pitch) is suspicious for synthesized audio.
    """
    try:
        f0 = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=512)
        # f0 may contain np.nan for unvoiced frames
        voiced = f0[~np.isnan(f0)]
        if voiced.size < 3:
            # not enough voiced content — return high variability to avoid false positive
            return 1.0
        mean_f0 = np.mean(voiced) + 1e-9
        var = np.std(voiced) / mean_f0
        return float(var)
    except Exception:
        # If pitch extraction fails, return neutral high variability (avoid false positives)
        return 1.0

def silence_gap_ratio(audio: np.ndarray, sr: int, top_db: int = 30) -> float:
    """
    Fraction of time that is silent (non-speech) using librosa.effects.split.
    Human speech tends to have natural micro-pauses; fully synthetic clips can be unnaturally continuous.
    Returns ratio of silence duration to total duration.
    """
    try:
        intervals = librosa.effects.split(audio, top_db=top_db)
        voiced_duration = sum((end - start) for start, end in intervals)
        total_len = len(audio)
        silence = max(0, total_len - voiced_duration)
        return float(silence / (total_len + 1e-12))
    except Exception:
        return 0.0

# ----------------------------
# Hybrid decision logic
# ----------------------------
def analyze_dsp_features(audio: np.ndarray, sr: int) -> Tuple[str, float, str]:
    """
    Compute multiple DSP features and combine them into a single suspiciousness score.
    Returns classification, confidence, and a human-readable message.
    """
    # Normalize amplitude to [-1,1]
    if np.max(np.abs(audio)) > 0:
        audio = audio / (np.max(np.abs(audio)) + 1e-12)

    # Short audio guard
    duration_sec = len(audio) / sr
    if duration_sec < 0.6:
        # Too short to be reliable; treat as unknown but lean human (to avoid false positives)
        return "HUMAN", 0.65, "Audio too short for firm verdict; treated as HUMAN (len < 0.6s)."

    std_dev = float(np.std(audio))
    zcr = zero_crossing_rate(audio)
    spec_ent = spectral_entropy(audio, sr)
    energy_jitter = rmse_energy_jitter(audio)
    pitch_var = pitch_variability(audio, sr)
    silence_ratio = silence_gap_ratio(audio, sr)

    # Compose weighted suspiciousness score (higher -> more likely synthetic)
    # Weights chosen conservatively to avoid false positives on quiet human audio
    w_std = 1.0     # very low std suggests smoothing
    w_zcr = 0.8
    w_spec = 1.2
    w_energy = 1.0
    w_pitch = 1.0
    w_silence = -0.6  # more silence means more natural pauses -> decrease synthetic score

    # Normalize features into 0..1 suspicious scale
    # Note: each transform chosen to map typical clean human values to low suspiciousness
    s_std = max(0.0, 1.0 - (std_dev / 0.12))          # std_dev < 0.12 contributes suspicion
    s_zcr = 0.0
    if zcr < 0.01:
        s_zcr = 0.9
    elif zcr > 0.18:
        s_zcr = 0.7
    else:
        s_zcr = 0.0

    s_spec = max(0.0, 1.0 - spec_ent)                 # low entropy -> suspicious
    s_energy = max(0.0, 1.0 - energy_jitter / 0.7)    # very low jitter -> suspicious
    s_pitch = max(0.0, 1.0 - pitch_var / 0.6)         # very low pitch variance suspicious
    s_silence = max(0.0, 1.0 - silence_ratio * 2.0)   # more silence -> reduce suspicion

    # raw score
    raw_score = (w_std*s_std + w_zcr*s_zcr + w_spec*s_spec + w_energy*s_energy + w_pitch*s_pitch + w_silence*s_silence)
    # Map raw_score to 0..1 by an activation-like transform
    score = 1.0 / (1.0 + math.exp(- (raw_score - 1.2)))  # sigmoid centered at 1.2

    # Decision thresholds
    if score > 0.65:
        classification = "VOICE_DEEPFAKE"
        confidence = min(0.99, 0.65 + (score - 0.65) * 0.9)
    else:
        classification = "HUMAN"
        confidence = min(0.99, 0.65 + (0.65 - score) * 0.5)

    message = (
        f"Score:{score:.3f} std:{std_dev:.4f} zcr:{zcr:.3f} spec_ent:{spec_ent:.3f} "
        f"energy_jitter:{energy_jitter:.3f} pitch_var:{pitch_var:.3f} silence:{silence_ratio:.3f}"
    )

    return classification, float(confidence), message

# ----------------------------
# Endpoint logic
# ----------------------------
@app.post("/analyze_voice", response_model=AnalysisResult)
async def analyze_voice_endpoint(audio_file: UploadFile = File(...)):
    contents = await audio_file.read()
    if not contents or len(contents) < 44:
        # 44 bytes is a tiny WAV header; treat as invalid
        raise HTTPException(status_code=400, detail="Empty or invalid audio file.")

    audio, sr = safe_load_audio(contents, target_sr=16000)
    classification, confidence, message = analyze_dsp_features(audio, sr)
    return AnalysisResult(classification=classification, confidence=confidence, message=message)

@app.get("/")
def read_root():
    return {"status": "Service Running", "message": "Use POST /analyze_voice with an audio file."}
