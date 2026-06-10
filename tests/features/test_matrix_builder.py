import pytest
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base, WorldCup, Match, Player, Country, Culture, Conflict
from src.features.matrix_builder import FeatureMatrixBuilder

@pytest.fixture(scope="module")
def test_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session(test_engine):
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

def test_build_matrix(db_session):
    # Insert mock data
    wc = WorldCup(year=2026, host="USA", num_teams=48)
    db_session.add(wc)
    
    country = Country(country_code="BRA", year=2026, political_stability=-0.2)
    db_session.add(country)
    
    culture = Culture(country_code="BRA", uai=76, trust=0.1)
    db_session.add(culture)
    
    player1 = Player(name="P1", country="Brazil", tournament_year=2026, adversity_score=8.0)
    player2 = Player(name="P2", country="Brazil", tournament_year=2026, adversity_score=2.0)
    db_session.add_all([player1, player2])
    
    db_session.commit()
    
    # Actually, we need to map "Brazil" (Player.country) to "BRA" (Country.country_code).
    # The matrix builder will handle this or we can just mock the team correctly.
    
    builder = FeatureMatrixBuilder(session_factory=lambda: db_session)
    # Testing matrix building
    # Since we lack some features, the builder should just use defaults or available ones
    matrix = builder.build(tournament_year=2026)
    
    assert isinstance(matrix, pd.DataFrame)
    # The matrix should have one row for Brazil
    # But wait, without Matches, how does it know Brazil qualified?
    # For now, it might scan Players.
    assert not matrix.empty
    assert "BRA" in matrix["country_code"].values
    
    # Check if features are assembled
    bra_row = matrix[matrix["country_code"] == "BRA"].iloc[0]
    assert bra_row["ppi"] == pytest.approx(0.27) # (1 - (-0.2 + 2.5)/5.0) * 0.5
    assert bra_row["adversity_mean"] == 5.0 # (8.0 + 2.0) / 2
