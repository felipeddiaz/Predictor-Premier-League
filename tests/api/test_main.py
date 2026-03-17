"""
Tests for the PL Predictor API.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.main import app
from api.schemas import MatchInputSchema

# Test client
client = TestClient(app)

class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "PL Predictor API"

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]

    def test_status_endpoint(self):
        """Test detailed status endpoint."""
        response = client.get("/status")
        assert response.status_code == 200
        data = response.json()
        assert "api" in data
        assert "predictor" in data
        assert data["api"] == "online"


class TestSimplePredictEndpoint:
    """Test the new simple /predict and /teams endpoints."""

    def test_teams_endpoint(self):
        """Test GET /teams returns team list."""
        response = client.get("/teams")
        assert response.status_code == 200
        teams = response.json()
        assert isinstance(teams, list)
        assert len(teams) == 20
        assert "Arsenal" in teams
        assert "Liverpool" in teams

    def test_simple_predict(self):
        """Test POST /predict with just team names."""
        response = client.post("/predict", json={
            "home_team": "Arsenal",
            "away_team": "Chelsea"
        })
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "home_team" in data
            assert "away_team" in data
            assert "home_win_probability" in data
            assert "draw_probability" in data
            assert "away_win_probability" in data
            assert "predicted_outcome" in data
            assert "confidence" in data
            assert 0 <= data["home_win_probability"] <= 1
            assert 0 <= data["draw_probability"] <= 1
            assert 0 <= data["away_win_probability"] <= 1
            assert data["predicted_outcome"] in ["Home Win", "Draw", "Away Win"]

    def test_simple_predict_missing_fields(self):
        """Test POST /predict with missing fields."""
        response = client.post("/predict", json={"home_team": "Arsenal"})
        assert response.status_code == 422


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    @pytest.fixture
    def valid_match_data(self):
        """Fixture with valid match data."""
        return {
            "local": "Manchester City",
            "visitante": "Liverpool",
            "cuota_h": 1.95,
            "cuota_d": 3.75,
            "cuota_a": 3.80
        }

    @pytest.fixture
    def match_with_ah(self):
        """Fixture with match data including Asian Handicap."""
        return {
            "local": "Chelsea",
            "visitante": "Arsenal",
            "cuota_h": 2.10,
            "cuota_d": 3.60,
            "cuota_a": 3.20,
            "ah_line": -1.5,
            "ah_cuota_h": 1.93,
            "ah_cuota_a": 1.97
        }

    def test_predict_match_basic(self, valid_match_data):
        """Test basic match prediction."""
        response = client.post("/predictions", json=valid_match_data)
        assert response.status_code in [200, 503]  # May fail if predictor not loaded

        if response.status_code == 200:
            data = response.json()
            assert len(data) >= 1
            prediction = data[0]
            assert "match" in prediction
            assert "prediction" in prediction
            assert "confidence" in prediction
            assert 0 <= prediction["confidence"] <= 1
            assert "prob_local" in prediction
            assert "prob_draw" in prediction
            assert "prob_away" in prediction

    def test_predict_match_with_ah(self, match_with_ah):
        """Test prediction with Asian Handicap data."""
        response = client.post("/predictions", json=match_with_ah)
        assert response.status_code in [200, 503]

    def test_predict_match_detail(self, valid_match_data):
        """Test detailed prediction endpoint."""
        response = client.post("/predictions/detail", json=valid_match_data)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            # Check main prediction fields
            assert "match" in data
            assert "prediction" in data
            assert "confidence" in data
            # Check probabilities
            assert "prob_local" in data
            assert "prob_draw" in data
            assert "prob_away" in data
            # Check market probabilities
            assert "market_prob_local" in data
            # Check binary markets
            assert "over25_prob" in data
            assert "over35_cards_prob" in data
            assert "over95_corners_prob" in data

    def test_invalid_teams(self):
        """Test prediction with invalid team names."""
        invalid_data = {
            "local": "",
            "visitante": "Liverpool",
            "cuota_h": 2.0,
            "cuota_d": 3.0,
            "cuota_a": 3.5
        }
        response = client.post("/predictions", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_invalid_odds(self):
        """Test prediction with invalid odds."""
        invalid_data = {
            "local": "Manchester City",
            "visitante": "Liverpool",
            "cuota_h": 0.5,  # Invalid: must be > 1.0
            "cuota_d": 3.0,
            "cuota_a": 3.5
        }
        response = client.post("/predictions", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_missing_required_fields(self):
        """Test prediction with missing required fields."""
        incomplete_data = {
            "local": "Manchester City",
            "visitante": "Liverpool"
            # Missing odds
        }
        response = client.post("/predictions", json=incomplete_data)
        assert response.status_code == 422  # Validation error


class TestSchemaValidation:
    """Test input schema validation."""

    def test_match_input_schema_valid(self):
        """Test valid MatchInputSchema."""
        data = {
            "local": "Arsenal",
            "visitante": "Chelsea",
            "cuota_h": 2.0,
            "cuota_d": 3.0,
            "cuota_a": 3.5
        }
        schema = MatchInputSchema(**data)
        assert schema.local == "Arsenal"
        assert schema.visitante == "Chelsea"
        assert schema.cuota_h == 2.0

    def test_match_input_schema_with_ah(self):
        """Test MatchInputSchema with Asian Handicap."""
        data = {
            "local": "Arsenal",
            "visitante": "Chelsea",
            "cuota_h": 2.0,
            "cuota_d": 3.0,
            "cuota_a": 3.5,
            "ah_line": -1.5,
            "ah_cuota_h": 1.90,
            "ah_cuota_a": 1.95
        }
        schema = MatchInputSchema(**data)
        assert schema.ah_line == -1.5

    def test_match_input_schema_invalid_odds(self):
        """Test invalid odds in schema."""
        data = {
            "local": "Arsenal",
            "visitante": "Chelsea",
            "cuota_h": 0.5,  # Invalid
            "cuota_d": 3.0,
            "cuota_a": 3.5
        }
        with pytest.raises(ValueError):
            MatchInputSchema(**data)


class TestAPIDocumentation:
    """Test API documentation generation."""

    def test_openapi_schema(self):
        """Test OpenAPI schema is generated."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "PL Predictor API"

    def test_swagger_ui(self):
        """Test Swagger UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc(self):
        """Test ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
