import pytest
import pandas as pd
from src.data.football_loader import FootballLoader
from src.database import SessionLocal, init_db, WorldCup, Match, Player

@pytest.fixture(scope="module")
def setup_db():
    init_db()
    yield SessionLocal

def test_football_loader_run(setup_db):
    loader = FootballLoader(setup_db)
    
    # Run the ETL process
    # Note: This makes real network requests. In a CI environment, we'd mock these.
    loader.run()
    
    session = setup_db()
    try:
        # 1. Verify World Cups
        wc_count = session.query(WorldCup).count()
        assert wc_count >= 22 # 1930-2022 + 2026 if placeholder added
        
        # 2. Verify Matches
        match_count = session.query(Match).count()
        assert match_count > 900
        
        # 3. Verify Players
        player_count = session.query(Player).count()
        assert player_count > 5000
        
        # Check specific historical data
        brazil_2002 = session.query(WorldCup).filter_by(year=2002).first()
        assert brazil_2002.winner == "Brazil"
        
        final_2022 = session.query(Match).filter_by(tournament_year=2022, stage="final").first()
        if final_2022:
            assert final_2022.home_team in ["Argentina", "France"]
            assert final_2022.away_team in ["Argentina", "France"]

    finally:
        session.close()

if __name__ == "__main__":
    pytest.main([__file__])
