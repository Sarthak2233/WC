import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base, Culture
from src.data.culture_loader import CultureLoader

@pytest.fixture(scope="module")
def test_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def loader(test_engine):
    Session = sessionmaker(bind=test_engine)
    return CultureLoader(session_factory=Session)

def test_extract_mock_data(loader):
    data = loader.extract()
    assert "culture" in data
    assert isinstance(data["culture"], pd.DataFrame)

def test_transform_data(loader):
    raw = {
        "culture": pd.DataFrame([
            {"country_code": "BRA", "pdi": 69, "idv": 38, "trust": 0.1}
        ])
    }
    transformed = loader.transform(raw)
    assert "culture" in transformed
    assert transformed["culture"].iloc[0]["pdi"] == 69
    assert transformed["culture"].iloc[0]["trust"] == 0.1

def test_load_data(loader, test_engine):
    df = pd.DataFrame([
        {"country_code": "FRA", "pdi": 68, "idv": 71, "trust": 0.22, "happiness_score": 6.6}
    ])
    
    loader.load({"culture": df})
    
    Session = sessionmaker(bind=test_engine)
    session = Session()
    culture = session.query(Culture).filter_by(country_code="FRA").first()
    assert culture is not None
    assert culture.pdi == 68
    assert culture.happiness_score == 6.6
    session.close()
