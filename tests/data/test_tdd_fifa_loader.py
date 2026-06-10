import pytest
import pandas as pd
from src.data.fifa_loader import FifaLoader
from src.fifa_database import PlayerRaw, FifaSessionLocal, init_fifa_db

@pytest.fixture(autouse=True)
def setup_db():
    init_fifa_db()

def test_fifa_loader_upsert():
    loader = FifaLoader(FifaSessionLocal)
    
    # Mock data: 1 new player, 1 existing player to update
    data = pd.DataFrame([
        {"full_name": "Test Player", "nationality": "Argentina", "overall": 80},
        {"full_name": "Lionel Messi", "nationality": "Argentina", "overall": 95}
    ])
    
    # Run load (populate first)
    loader.load(data)
    
    # Verify population
    session = FifaSessionLocal()
    player = session.query(PlayerRaw).filter_by(full_name="Test Player").first()
    assert player is not None
    assert player.overall == 80
    session.close()
    
    # Run load (update)
    update_data = pd.DataFrame([
        {"full_name": "Lionel Messi", "nationality": "Argentina", "overall": 99}
    ])
    loader.load(update_data)
    
    # Verify update
    session = FifaSessionLocal()
    messi = session.query(PlayerRaw).filter_by(full_name="Lionel Messi").first()
    assert messi.overall == 99
    session.close()
