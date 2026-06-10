import pytest
from unittest.mock import MagicMock, patch
from src.data.conflict_loader import ConflictLoader
import os

# Mock session factory
mock_session_factory = MagicMock()

@patch('src.data.conflict_loader.requests.post')
@patch('src.data.conflict_loader.requests.get')
def test_fetch_acled_oauth_flow(mock_get, mock_post, monkeypatch):
    # Set env vars for testing
    monkeypatch.setenv("ACLED_EMAIL", "test@test.com")
    monkeypatch.setenv("ACLED_PASSWORD", "password123")
    
    # Mock token response
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "fake_token_123"}
    
    # Mock API response
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"success": True, "data": []}
    
    loader = ConflictLoader(mock_session_factory)
    
    # Run fetch
    data = loader._fetch_acled()
    
    # Assertions
    mock_post.assert_called_once()
    # Verify Authorization header in the get request
    args, kwargs = mock_get.call_args
    assert "Authorization" in kwargs["headers"]
    assert kwargs["headers"]["Authorization"] == "Bearer fake_token_123"
