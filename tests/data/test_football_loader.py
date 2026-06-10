import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base, WorldCup
from src.data.football_loader import FootballLoader

@pytest.fixture(scope="module")
def test_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def loader(test_engine):
    Session = sessionmaker(bind=test_engine)
    return FootballLoader(session_factory=Session)

def test_extract_mock_data(loader):
    """Test the extraction method logic."""
    data = loader.extract()
    assert "world_cups" in data
    assert isinstance(data["world_cups"], pd.DataFrame)

def test_transform_data(loader):
    """Test data transformation and standardization."""
    raw = {
        "world_cups": pd.DataFrame([
            {"Year": 2026, "Host": "USA, Canada, Mexico", "Teams": 48}
        ])
    }
    transformed = loader.transform(raw)
    assert "world_cups" in transformed
    assert transformed["world_cups"].iloc[0]["year"] == 2026

def test_load_data(loader, test_engine):
    """Test database load."""
    df = pd.DataFrame([
        {"year": 2026, "host": "United States/Canada/Mexico", "num_teams": 48}
    ])
    
    loader.load({"world_cups": df})
    
    Session = sessionmaker(bind=test_engine)
    session = Session()
    wc = session.query(WorldCup).filter_by(year=2026).first()
    assert wc is not None
    assert wc.num_teams == 48
    session.close()
