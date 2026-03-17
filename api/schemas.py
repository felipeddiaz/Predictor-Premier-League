from pydantic import BaseModel, Field
from typing import Optional

# ============================================================================
# INPUT SCHEMAS
# ============================================================================

class SimpleMatchInput(BaseModel):
    """Schema for simple match prediction — only team names required."""
    home_team: str = Field(..., description="Home team name (e.g. 'Arsenal')")
    away_team: str = Field(..., description="Away team name (e.g. 'Chelsea')")

    class Config:
        json_schema_extra = {
            "example": {
                "home_team": "Arsenal",
                "away_team": "Chelsea"
            }
        }

class MatchInputSchema(BaseModel):
    """Schema for detailed match prediction input (with betting odds)."""
    local: str = Field(..., description="Nombre del equipo local")
    visitante: str = Field(..., description="Nombre del equipo visitante")
    cuota_h: float = Field(..., gt=1.0, description="Cuota de victoria local (ej: 2.10)")
    cuota_d: float = Field(..., gt=1.0, description="Cuota de empate (ej: 3.40)")
    cuota_a: float = Field(..., gt=1.0, description="Cuota de victoria visitante (ej: 3.20)")
    # Optional Asian Handicap
    ah_line: Optional[float] = Field(None, description="Línea de Asian Handicap (ej: -1.5)")
    ah_cuota_h: Optional[float] = Field(None, gt=1.0, description="Cuota AH para local")
    ah_cuota_a: Optional[float] = Field(None, gt=1.0, description="Cuota AH para visitante")

    class Config:
        json_schema_extra = {
            "example": {
                "local": "Manchester City",
                "visitante": "Liverpool",
                "cuota_h": 1.95,
                "cuota_d": 3.75,
                "cuota_a": 3.80,
                "ah_line": None,
                "ah_cuota_h": None,
                "ah_cuota_a": None
            }
        }

# ============================================================================
# OUTPUT SCHEMAS
# ============================================================================

class SimplePredictionResponse(BaseModel):
    """Schema for the simple /predict endpoint — user-friendly output."""
    home_team: str
    away_team: str
    home_win_probability: float = Field(..., ge=0, le=1)
    draw_probability: float = Field(..., ge=0, le=1)
    away_win_probability: float = Field(..., ge=0, le=1)
    predicted_outcome: str = Field(..., description="'Home Win', 'Draw', or 'Away Win'")
    confidence: float = Field(..., ge=0, le=1)
    home_form: str = Field(..., description="Recent form e.g. '3W-1D-1L'")
    away_form: str = Field(..., description="Recent form e.g. '2W-2D-1L'")

    class Config:
        json_schema_extra = {
            "example": {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_win_probability": 0.52,
                "draw_probability": 0.23,
                "away_win_probability": 0.25,
                "predicted_outcome": "Home Win",
                "confidence": 0.52,
                "home_form": "3W-1D-1L",
                "away_form": "2W-2D-1L"
            }
        }

class PredictionResponse(BaseModel):
    """Schema for basic prediction response."""
    match: str
    prediction: str
    confidence: float = Field(..., ge=0, le=1, description="Probabilidad de la predicción")
    prob_local: float = Field(..., ge=0, le=1)
    prob_draw: float = Field(..., ge=0, le=1)
    prob_away: float = Field(..., ge=0, le=1)
    edge: float = Field(..., description="Diferencia de valor (edge)")
    form_local: str = Field(..., description="Forma reciente del equipo local")
    form_away: str = Field(..., description="Forma reciente del equipo visitante")

    class Config:
        json_schema_extra = {
            "example": {
                "match": "Manchester City vs Liverpool",
                "prediction": "Local",
                "confidence": 0.72,
                "prob_local": 0.72,
                "prob_draw": 0.18,
                "prob_away": 0.10,
                "edge": 0.08,
                "form_local": "3W-1D-1L",
                "form_away": "2W-2D-1L"
            }
        }

class PredictionDetailResponse(BaseModel):
    """Schema for detailed prediction including binary markets."""
    match: str
    local: str
    away: str
    prediction: str
    confidence: float = Field(..., ge=0, le=1)
    # Model probabilities
    prob_local: float = Field(..., ge=0, le=1)
    prob_draw: float = Field(..., ge=0, le=1)
    prob_away: float = Field(..., ge=0, le=1)
    # Market probabilities
    market_prob_local: float = Field(..., ge=0, le=1)
    market_prob_draw: float = Field(..., ge=0, le=1)
    market_prob_away: float = Field(..., ge=0, le=1)
    # Edge and form
    edge: float
    form_local: str
    form_away: str
    # Binary markets (Over/Under, Cards, Corners)
    over25_prob: Optional[float] = Field(None, ge=0, le=1, description="Probabilidad Over 2.5 goles")
    over35_cards_prob: Optional[float] = Field(None, ge=0, le=1, description="Probabilidad Over 3.5 tarjetas")
    over95_corners_prob: Optional[float] = Field(None, ge=0, le=1, description="Probabilidad Over 9.5 corners")

    class Config:
        json_schema_extra = {
            "example": {
                "match": "Manchester City vs Liverpool",
                "local": "Manchester City",
                "away": "Liverpool",
                "prediction": "Local",
                "confidence": 0.72,
                "prob_local": 0.72,
                "prob_draw": 0.18,
                "prob_away": 0.10,
                "market_prob_local": 0.64,
                "market_prob_draw": 0.25,
                "market_prob_away": 0.11,
                "edge": 0.08,
                "form_local": "3W-1D-1L",
                "form_away": "2W-2D-1L",
                "over25_prob": 0.65,
                "over35_cards_prob": 0.45,
                "over95_corners_prob": 0.58
            }
        }

class JornadaPredictionResponse(BaseModel):
    """Schema for a full gameweek predictions."""
    gameweek: int
    predictions: list[PredictionResponse]
    total_matches: int

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Estado de la API: healthy o degraded")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy"
            }
        }
