from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.predictor import Predictor
from core.models import Partido, ConfigJornada
from api.schemas import (
    PredictionResponse,
    PredictionDetailResponse,
    JornadaPredictionResponse,
    MatchInputSchema,
    HealthResponse
)
from api.routers import (
    router_gameweek,
    router_teams,
    router_history,
    router_utils,
    set_predictor_instance
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PL Predictor API",
    description="API para predicciones de la Premier League usando ML",
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

# Include routers
app.include_router(router_gameweek)
app.include_router(router_teams)
app.include_router(router_history)
app.include_router(router_utils)

# Global predictor instance
predictor = None

@app.on_event("startup")
async def startup_event():
    """Load predictor on startup."""
    global predictor
    try:
        predictor = Predictor()
        if predictor.cargar():
            logger.info("✓ Predictor cargado exitosamente")
            # Share predictor instance with routers
            set_predictor_instance(predictor)
        else:
            logger.warning("⚠ Error al cargar el predictor")
    except Exception as e:
        logger.error(f"✗ Error durante startup: {e}")
        predictor = None

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PL Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "online",
        "predictor_loaded": predictor is not None
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy" if predictor else "degraded")

@app.post("/predictions", response_model=list[PredictionResponse])
async def predict_match(match: MatchInputSchema):
    """
    Predict a single Premier League match.

    Provides:
    - Win/Draw/Loss probabilities
    - Model prediction and confidence
    - Market value (edge)
    - Team form
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    try:
        # Create Partido object from input
        partido = Partido(
            local=match.local,
            visitante=match.visitante,
            cuota_h=match.cuota_h,
            cuota_d=match.cuota_d,
            cuota_a=match.cuota_a,
            ah_line=match.ah_line,
            ah_cuota_h=match.ah_cuota_h,
            ah_cuota_a=match.ah_cuota_a
        )

        # Get prediction from model
        prediccion = predictor.predecir_partido(partido)

        if prediccion is None:
            raise HTTPException(status_code=400, detail="No se pudo generar predicción")

        # Format response
        return [PredictionResponse(
            match=f"{prediccion.partido.local} vs {prediccion.partido.visitante}",
            prediction=prediccion.resultado_predicho,
            confidence=prediccion.confianza,
            prob_local=prediccion.prob_local,
            prob_draw=prediccion.prob_empate,
            prob_away=prediccion.prob_visitante,
            edge=prediccion.diferencia_valor,
            form_local=prediccion.forma_local,
            form_away=prediccion.forma_visitante
        )]

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/predictions/detail", response_model=PredictionDetailResponse)
async def predict_match_detail(match: MatchInputSchema):
    """
    Get detailed prediction including binary markets (Over/Under, Cards, Corners).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    try:
        partido = Partido(
            local=match.local,
            visitante=match.visitante,
            cuota_h=match.cuota_h,
            cuota_d=match.cuota_d,
            cuota_a=match.cuota_a,
            ah_line=match.ah_line,
            ah_cuota_h=match.ah_cuota_h,
            ah_cuota_a=match.ah_cuota_a
        )

        # Main prediction
        prediccion = predictor.predecir_partido(partido)
        if prediccion is None:
            raise HTTPException(status_code=400, detail="No se pudo generar predicción")

        # Binary markets prediction
        prediccion_binaria = predictor.predecir_mercados_binarios(partido)

        return PredictionDetailResponse(
            match=f"{prediccion.partido.local} vs {prediccion.partido.visitante}",
            local=prediccion.partido.local,
            away=prediccion.partido.visitante,
            prediction=prediccion.resultado_predicho,
            confidence=prediccion.confianza,
            # Main probabilities
            prob_local=prediccion.prob_local,
            prob_draw=prediccion.prob_empate,
            prob_away=prediccion.prob_visitante,
            # Market probabilities
            market_prob_local=prediccion.prob_mercado_local,
            market_prob_draw=prediccion.prob_mercado_empate,
            market_prob_away=prediccion.prob_mercado_visitante,
            # Edge
            edge=prediccion.diferencia_valor,
            # Form
            form_local=prediccion.forma_local,
            form_away=prediccion.forma_visitante,
            # Binary markets
            over25_prob=prediccion_binaria.prob_over25,
            over35_cards_prob=prediccion_binaria.prob_over35_cards,
            over95_corners_prob=prediccion_binaria.prob_over95_corners
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción detallada: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/status", response_model=dict)
async def get_status():
    """Get detailed API and predictor status."""
    status = {
        "api": "online",
        "predictor": "loaded" if predictor else "not_loaded",
        "version": "1.0.0"
    }

    if predictor and hasattr(predictor, 'modelo_principal'):
        status["models"] = {
            "main": "loaded" if predictor.modelo_principal else "not_loaded",
            "vb": "loaded" if hasattr(predictor, 'modelo_vb') and predictor.modelo_vb else "not_loaded",
            "ou": "loaded" if hasattr(predictor, 'modelo_ou') and predictor.modelo_ou else "not_loaded"
        }

    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
