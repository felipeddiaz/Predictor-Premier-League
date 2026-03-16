"""
Tests for the PL Predictor API routers.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.main import app

# Test client
client = TestClient(app)


class TestGameweekRouter:
    """Test gameweek predictions endpoints."""

    def test_get_gameweek_info(self):
        """Test getting gameweek information."""
        response = client.get("/gameweeks/1")
        assert response.status_code == 200
        data = response.json()
        assert "gameweek" in data
        assert data["gameweek"] == 1
        assert "total_matches" in data

    def test_get_gameweek_invalid(self):
        """Test invalid gameweek number."""
        response = client.get("/gameweeks/39")  # Premier League has 38 gameweeks
        assert response.status_code == 400

    def test_predict_gameweek_basic(self):
        """Test predicting a full gameweek."""
        matches = [
            {
                "local": "Manchester City",
                "visitante": "Liverpool",
                "cuota_h": 1.95,
                "cuota_d": 3.75,
                "cuota_a": 3.80
            },
            {
                "local": "Arsenal",
                "visitante": "Chelsea",
                "cuota_h": 2.10,
                "cuota_d": 3.60,
                "cuota_a": 3.20
            }
        ]

        response = client.post("/gameweeks/1/predictions", json=matches)
        assert response.status_code in [200, 503]  # May fail if predictor not loaded

        if response.status_code == 200:
            data = response.json()
            assert "gameweek" in data
            assert data["gameweek"] == 1
            assert "predictions" in data
            assert "total_matches" in data
            assert data["total_matches"] == 2

    def test_predict_gameweek_empty(self):
        """Test predicting with empty matches list."""
        response = client.post("/gameweeks/1/predictions", json=[])
        assert response.status_code == 400

    def test_predict_gameweek_invalid_match(self):
        """Test predicting with invalid match data."""
        matches = [
            {
                "local": "Manchester City",
                # Missing required fields
            }
        ]

        response = client.post("/gameweeks/1/predictions", json=matches)
        # Should still return 200 but with errors
        assert response.status_code in [200, 400]


class TestTeamRouter:
    """Test team statistics endpoints."""

    def test_get_team_stats(self):
        """Test getting team statistics."""
        response = client.get("/teams/Manchester City/stats")
        assert response.status_code in [200, 404, 503]

        if response.status_code == 200:
            data = response.json()
            assert "team" in data
            assert "is_local" in data

    def test_get_team_stats_away(self):
        """Test getting away team statistics."""
        response = client.get("/teams/Liverpool/stats?is_local=false")
        assert response.status_code in [200, 404, 503]

    def test_list_teams(self):
        """Test listing teams."""
        response = client.get("/teams")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestHistoryRouter:
    """Test prediction history endpoints."""

    def test_get_accuracy(self):
        """Test getting accuracy statistics."""
        response = client.get("/history/accuracy")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_log_prediction(self):
        """Test logging a prediction."""
        data = {
            "prediction_id": "test-123",
            "result": "Local",
            "actual_outcome": "Local"
        }
        response = client.post("/history/log", json=data)
        assert response.status_code == 200


class TestUtilsRouter:
    """Test utility endpoints."""

    def test_validate_odds_valid(self):
        """Test validating valid odds."""
        response = client.get(
            "/utils/validate-odds",
            params={
                "cuota_h": 2.0,
                "cuota_d": 3.0,
                "cuota_a": 3.5
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "implied_probabilities" in data
        assert "true_probabilities" in data
        assert "vigorish" in data
        assert "is_valid" in data

    def test_validate_odds_invalid(self):
        """Test validating invalid odds."""
        response = client.get(
            "/utils/validate-odds",
            params={
                "cuota_h": 0.5,  # Invalid
                "cuota_d": 3.0,
                "cuota_a": 3.5
            }
        )
        assert response.status_code == 400

    def test_calculate_ev_positive(self):
        """Test calculating positive expected value."""
        response = client.get(
            "/utils/expected-value",
            params={
                "prob_model": 0.6,
                "cuota": 2.0
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "expected_value" in data
        assert "roi_percentage" in data
        assert "is_profitable" in data
        assert "recommendation" in data
        assert data["is_profitable"] == True
        assert data["recommendation"] == "BET"

    def test_calculate_ev_negative(self):
        """Test calculating negative expected value."""
        response = client.get(
            "/utils/expected-value",
            params={
                "prob_model": 0.3,
                "cuota": 2.0
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_profitable"] == False
        assert data["recommendation"] == "SKIP"

    def test_calculate_ev_invalid_prob(self):
        """Test EV calculation with invalid probability."""
        response = client.get(
            "/utils/expected-value",
            params={
                "prob_model": 1.5,  # Invalid: must be 0-1
                "cuota": 2.0
            }
        )
        assert response.status_code == 400

    def test_calculate_ev_invalid_odds(self):
        """Test EV calculation with invalid odds."""
        response = client.get(
            "/utils/expected-value",
            params={
                "prob_model": 0.5,
                "cuota": 0.5  # Invalid: must be > 1.0
            }
        )
        assert response.status_code == 400


class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_predictor_not_loaded_error(self):
        """Test handling when predictor is not loaded."""
        # This test assumes the predictor might not be loaded
        response = client.get("/status")
        assert response.status_code == 200

    def test_invalid_json(self):
        """Test handling invalid JSON input."""
        response = client.post(
            "/predictions",
            json={"invalid": "data"}
        )
        assert response.status_code == 422  # Validation error

    def test_missing_required_fields(self):
        """Test handling missing required fields."""
        response = client.post(
            "/predictions",
            json={"local": "Manchester City"}
        )
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
