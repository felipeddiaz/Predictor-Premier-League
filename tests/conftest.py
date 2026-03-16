"""
Pytest configuration and fixtures for all tests.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get the test data directory."""
    data_dir = project_root / "tests" / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def valid_match_data():
    """Fixture with valid match data for testing."""
    return {
        "local": "Manchester City",
        "visitante": "Liverpool",
        "cuota_h": 1.95,
        "cuota_d": 3.75,
        "cuota_a": 3.80
    }


@pytest.fixture
def invalid_match_data():
    """Fixture with invalid match data for testing."""
    return {
        "local": "",  # Invalid: empty
        "visitante": "Liverpool",
        "cuota_h": 0.5,  # Invalid: < 1.0
        "cuota_d": 3.75,
        "cuota_a": 3.80
    }


@pytest.fixture
def sample_gameweek_matches():
    """Fixture with sample matches for a gameweek."""
    return [
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
        },
        {
            "local": "Manchester United",
            "visitante": "Tottenham",
            "cuota_h": 2.30,
            "cuota_d": 3.40,
            "cuota_a": 2.95
        }
    ]


# Markers for test categorization
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as a slow test"
    )
