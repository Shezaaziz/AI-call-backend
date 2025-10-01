import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

# --- 1. FastAPI Application Initialization ---
# Create the FastAPI app instance
app = FastAPI(
    title="AI Call Screening Service",
    description="Backend for real-time reputation analysis of incoming call numbers."
)

# --- NEW: Root Endpoint for Health Check ---
@app.get("/")
def read_root():
    return {"status": "Service Running", "message": "Access /docs for the API documentation."}
# ---------------------------------------------


# --- 2. Request and Response Schemas (Pydantic Models) ---

# Define the data structure for the incoming request from the Kotlin service
class NumberAnalysisRequest(BaseModel):
    # The phone number to be analyzed, including country code (e.g., +15551234567)
    phone_number: str

# Define the data structure for the response sent back to the Kotlin service
# The result field uses Literal to restrict possible classification values
class AnalysisResult(BaseModel):
    # Classification: NORMAL, SPAM, or AI_SCAM
    classification: Literal["NORMAL", "SPAM", "AI_SCAM"]
    # Confidence score (0.0 to 1.0)
    confidence: float
    # An optional human-readable message for logging
    message: str


# --- 3. Core AI Analysis Function (Simulated) ---

def run_reputation_analysis(number: str) -> str:
    """
    Simulates the AI model lookup based on number reputation.
    
    In a production system, this function would:
    1. Query a Firestore/Redis database for known spam lists.
    2. Access a trained model (e.g., a simple scikit-learn model or a larger ML service)
       that scores the number based on calling patterns, frequency, and carrier data.
    """
    
    # Simulate high-confidence AI Scam detection (e.g., known high-volume pattern)
    if number.endswith("9999"):
        return "AI_SCAM"
    
    # Simulate general spam/robocall detection (e.g., known marketing line)
    if number.endswith("5555"):
        return "SPAM"
        
    # Simulate a verified number or normal user
    return "NORMAL"

# --- 4. API Endpoint Definition ---

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_number(request: NumberAnalysisRequest):
    """
    Analyzes an incoming phone number and returns a classification.
    """
    
    # 🚨 Crucial: Simulate processing time. Keep this under 2 seconds!
    time.sleep(0.5) 

    number = request.phone_number
    
    # Get the simulated AI classification
    classification = run_reputation_analysis(number)
    
    # Set confidence and message based on classification
    if classification == "AI_SCAM":
        confidence = 0.95
        message = "High-confidence detection: Number matches known AI-generated robocall pattern."
    elif classification == "SPAM":
        confidence = 0.75
        message = "Medium-confidence detection: Number is flagged as general commercial spam."
    else:
        confidence = 0.50
        message = "Number appears normal or verified."

    # Return the structured result
    return AnalysisResult(
        classification=classification,
        confidence=confidence,
        message=message
    )
