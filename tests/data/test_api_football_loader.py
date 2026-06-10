import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.data.api_football_loader import ApiFootballLoader

# Mock API responses
mock_fixtures = {"response": [{"fixture": {"id": 1}, "teams": {"home": {"name": "USA"}, "away": {"name": "Mexico"}}}]}
mock_players = {"response": [{"player": {"name": "Lionel Messi"}, "statistics": [{"games": {"position": "Forward"}}]}], "paging": {"current": 1, "total": 1}}

@pytest.fixture
def mock_loader():
    loader = ApiFootballLoader(None)
    return loader

def test_fetch_json_handles_403_429(mock_loader, monkeypatch):
    # Mock requests.get
    mock_get = MagicMock()
    mock_get.status_code = 403
    monkeypatch.setattr("requests.get", MagicMock(return_value=mock_get))
    
    result = mock_loader._fetch_json("/test", {})
    assert result is None
    
    # Mock requests.get for 429
    mock_get.status_code = 429
    monkeypatch.setattr("requests.get", MagicMock(return_value=mock_get))
    
    result = mock_loader._fetch_json("/test", {})
    assert result is None
