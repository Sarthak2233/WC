import pytest
import pandas as pd
from src.data.football_loader import FootballLoader

@pytest.fixture
def loader():
    return FootballLoader()

def test_extract_mock_data(loader):
    """Test the extraction method logic."""
    data = loader.extract()
    assert "world_cups" in data
    assert isinstance(data["world_cups"], pd.DataFrame)

def test_transform_data(loader):
    """Test data transformation and standardization."""
    # Assuming the loader expects specific columns based on what it calls _fetch_csv
    raw = {
        "world_cups": pd.DataFrame([
            {"tournament_id": "WC-2026", "host_country": "USA", "winner": "USA"}
        ])
    }
    transformed = loader.transform(raw)
    assert "world_cups" in transformed
    assert transformed["world_cups"].iloc[0]["year"] == 2026
