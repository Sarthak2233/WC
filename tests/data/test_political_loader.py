import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base, Country
from src.data.political_loader import PoliticalLoader

@pytest.fixture(scope="module")
def test_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def loader(test_engine):
    Session = sessionmaker(bind=test_engine)
    return PoliticalLoader(session_factory=Session)

def test_extract_mock_data(loader):
    data = loader.extract()
    assert "countries" in data
    assert isinstance(data["countries"], pd.DataFrame)

def test_transform_data(loader):
    raw = {
        "countries": pd.DataFrame([
            {"country_code": "USA", "year": 2024, "political_stability": 0.8}
        ])
    }
    transformed = loader.transform(raw)
    assert "countries" in transformed
    assert transformed["countries"].iloc[0]["political_stability"] == 0.8

def test_load_data(loader, test_engine):
    df = pd.DataFrame([
        {"country_code": "BRA", "year": 2024, "political_stability": -0.2, "gdp_per_capita": 10000.0}
    ])
    
    loader.load({"countries": df})
    
    Session = sessionmaker(bind=test_engine)
    session = Session()
    country = session.query(Country).filter_by(country_code="BRA", year=2024).first()
    assert country is not None
    assert country.political_stability == -0.2
    assert country.gdp_per_capita == 10000.0
    session.close()
