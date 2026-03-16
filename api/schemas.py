from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    id: int
    match: str
    prediction: str
    confidence: float
    date: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "match": "Manchester City vs Liverpool",
                "prediction": "Manchester City Win",
                "confidence": 0.75,
                "date": "2024-03-23"
            }
        }

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
