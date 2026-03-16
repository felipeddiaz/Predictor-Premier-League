"""
Additional routers for the PL Predictor API.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, list
import logging

logger = logging.getLogger(__name__)

# Create routers
router_gameweek = APIRouter(prefix="/gameweeks", tags=["Gameweek Predictions"])
router_teams = APIRouter(prefix="/teams", tags=["Team Statistics"])
router_history = APIRouter(prefix="/history", tags=["Prediction History"])

predictor = None

# ============================================================================
# GAMEWEEK PREDICTIONS ROUTER
# ============================================================================

@router_gameweek.post("/{gameweek}/predictions")
async def predict_gameweek(gameweek: int, matches: list[dict]):
    """
    Predict all matches for a specific gameweek.

    Input: List of match dictionaries with team names and odds.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    if gameweek < 1 or gameweek > 38:
        raise HTTPException(status_code=400, detail="Gameweek must be between 1 and 38")

    if not matches or len(matches) == 0:
        raise HTTPException(status_code=400, detail="At least one match required")

    predictions = []
    errors = []

    for i, match_data in enumerate(matches):
        try:
            from core.models import Partido
            from api.schemas import PredictionResponse

            # Validate match data
            required_fields = ["local", "visitante", "cuota_h", "cuota_d", "cuota_a"]
            if not all(field in match_data for field in required_fields):
                errors.append(f"Match {i}: Missing required fields {required_fields}")
                continue

            # Create partido and predict
            partido = Partido(
                local=match_data["local"],
                visitante=match_data["visitante"],
                cuota_h=match_data["cuota_h"],
                cuota_d=match_data["cuota_d"],
                cuota_a=match_data["cuota_a"],
                ah_line=match_data.get("ah_line"),
                ah_cuota_h=match_data.get("ah_cuota_h"),
                ah_cuota_a=match_data.get("ah_cuota_a")
            )

            prediccion = predictor.predecir_partido(partido)

            if prediccion:
                predictions.append(PredictionResponse(
                    match=f"{prediccion.partido.local} vs {prediccion.partido.visitante}",
                    prediction=prediccion.resultado_predicho,
                    confidence=prediccion.confianza,
                    prob_local=prediccion.prob_local,
                    prob_draw=prediccion.prob_empate,
                    prob_away=prediccion.prob_visitante,
                    edge=prediccion.diferencia_valor,
                    form_local=prediccion.forma_local,
                    form_away=prediccion.forma_visitante
                ))

        except ValueError as e:
            errors.append(f"Match {i}: {str(e)}")
        except Exception as e:
            logger.error(f"Error predicting match {i}: {e}")
            errors.append(f"Match {i}: Internal error")

    return {
        "gameweek": gameweek,
        "total_matches": len(matches),
        "predictions": predictions,
        "successful": len(predictions),
        "errors": errors if errors else None
    }

@router_gameweek.get("/{gameweek}")
async def get_gameweek_info(gameweek: int):
    """Get information about a specific gameweek."""
    if gameweek < 1 or gameweek > 38:
        raise HTTPException(status_code=400, detail="Gameweek must be between 1 and 38")

    return {
        "gameweek": gameweek,
        "total_matches": 10,  # Fixed for Premier League
        "description": f"Premier League Gameweek {gameweek}"
    }

# ============================================================================
# TEAM STATISTICS ROUTER
# ============================================================================

@router_teams.get("/{team_name}/stats")
async def get_team_stats(team_name: str, is_local: Optional[bool] = True):
    """
    Get team statistics from the predictor.

    Parameters:
        team_name: Name of the team
        is_local: True for home stats, False for away stats
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    try:
        stats = predictor._obtener_stats_equipo(team_name, is_local)

        if stats is None:
            raise HTTPException(
                status_code=404,
                detail=f"Team '{team_name}' not found or no statistics available"
            )

        return {
            "team": team_name,
            "is_local": is_local,
            "stats": stats
        }

    except Exception as e:
        logger.error(f"Error fetching stats for {team_name}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching team statistics")

@router_teams.get("")
async def list_teams():
    """List all teams available for predictions."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    # This would require access to the teams from historical data
    # For now, return a message indicating the feature
    return {
        "message": "Use /teams/{team_name}/stats to get statistics for a specific team",
        "example": "GET /teams/Manchester City/stats?is_local=true"
    }

# ============================================================================
# PREDICTION HISTORY ROUTER
# ============================================================================

@router_history.get("/accuracy")
async def get_accuracy():
    """Get prediction accuracy statistics."""
    # This would require storing predictions in a database
    return {
        "message": "Accuracy tracking requires database integration",
        "note": "This feature will be available after adding a database"
    }

@router_history.post("/log")
async def log_prediction(prediction_id: str, result: str, actual_outcome: str):
    """
    Log a prediction result for accuracy tracking.

    Parameters:
        prediction_id: Unique ID of the prediction
        result: The predicted outcome
        actual_outcome: The actual match outcome
    """
    # This would require database integration
    return {
        "message": "Prediction logged successfully",
        "prediction_id": prediction_id,
        "note": "Persistence requires database integration"
    }

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

router_utils = APIRouter(prefix="/utils", tags=["Utilities"])

@router_utils.post("/validate-odds")
async def validate_odds(cuota_h: float, cuota_d: float, cuota_a: float):
    """
    Validate betting odds and calculate implied probabilities.

    Parameters:
        cuota_h: Home odds
        cuota_d: Draw odds
        cuota_a: Away odds
    """
    if cuota_h <= 1.0 or cuota_d <= 1.0 or cuota_a <= 1.0:
        raise HTTPException(status_code=400, detail="All odds must be > 1.0")

    # Calculate implied probabilities
    prob_h = 1 / cuota_h
    prob_d = 1 / cuota_d
    prob_a = 1 / cuota_a

    # Calculate vigorish (margin)
    total_prob = prob_h + prob_d + prob_a
    vigorish = total_prob - 1

    # True probabilities (removing vigorish)
    true_prob_h = prob_h / total_prob
    true_prob_d = prob_d / total_prob
    true_prob_a = prob_a / total_prob

    return {
        "input": {
            "cuota_h": cuota_h,
            "cuota_d": cuota_d,
            "cuota_a": cuota_a
        },
        "implied_probabilities": {
            "home": round(prob_h, 4),
            "draw": round(prob_d, 4),
            "away": round(prob_a, 4),
            "total": round(total_prob, 4)
        },
        "true_probabilities": {
            "home": round(true_prob_h, 4),
            "draw": round(true_prob_d, 4),
            "away": round(true_prob_a, 4)
        },
        "vigorish": {
            "value": round(vigorish, 4),
            "percentage": round(vigorish * 100, 2)
        },
        "is_valid": total_prob > 0.99  # Allow small floating point errors
    }

@router_utils.get("/expected-value")
async def calculate_ev(prob_model: float, cuota: float):
    """
    Calculate expected value of a bet.

    Parameters:
        prob_model: Probability according to the model (0-1)
        cuota: Betting odds
    """
    if not 0 <= prob_model <= 1:
        raise HTTPException(status_code=400, detail="Probability must be between 0 and 1")

    if cuota <= 1.0:
        raise HTTPException(status_code=400, detail="Odds must be > 1.0")

    # EV = (Probability × Profit) + ((1 - Probability) × Loss)
    # = Probability × (Odds - 1) - (1 - Probability)
    # = (Probability × Odds) - 1
    expected_value = (prob_model * cuota) - 1
    roi = (expected_value / 1) * 100 if expected_value != 0 else 0

    market_prob = 1 / cuota
    edge = prob_model - market_prob

    return {
        "model_probability": prob_model,
        "odds": cuota,
        "market_probability": round(market_prob, 4),
        "edge": round(edge, 4),
        "expected_value": round(expected_value, 4),
        "roi_percentage": round(roi, 2),
        "is_profitable": expected_value > 0,
        "recommendation": "BET" if expected_value > 0 else "SKIP"
    }

def set_predictor_instance(pred_instance):
    """Set the global predictor instance for routers."""
    global predictor
    predictor = pred_instance
