from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="PL Predictor API",
    description="API para predicciones de la Premier League",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "PL Predictor API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/predictions")
async def get_predictions():
    """
    Get current predictions for upcoming Premier League matches.

    This endpoint returns a list of predictions based on the ML model.
    """
    # TODO: Integrate with the existing Predictor model
    return [
        {
            "id": 1,
            "match": "Manchester City vs Liverpool",
            "prediction": "Manchester City Win",
            "confidence": 0.72,
            "date": "2024-03-23"
        },
        {
            "id": 2,
            "match": "Arsenal vs Chelsea",
            "prediction": "Draw",
            "confidence": 0.58,
            "date": "2024-03-23"
        }
    ]

@app.get("/predictions/{match_id}")
async def get_prediction(match_id: int):
    """Get a specific prediction by match ID."""
    return {
        "id": match_id,
        "match": "Sample Match",
        "prediction": "Sample Prediction",
        "confidence": 0.65
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
