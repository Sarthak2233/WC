import pytest
from src.data.political_loader import PoliticalLoader
from src.database import SessionLocal, init_db, Country

@pytest.fixture(scope="module")
def setup_db():
    init_db()
    yield SessionLocal

def test_political_loader_run(setup_db):
    loader = PoliticalLoader(setup_db)
    
    # Run ETL
    # Note: This fetches from World Bank API
    loader.run()
    
    session = setup_db()
    try:
        # Check if we have data for a major country
        usa_data = session.query(Country).filter_by(country_code="USA", year=2022).first()
        assert usa_data is not None
        assert usa_data.gdp_per_capita > 0
        
        # Check count
        count = session.query(Country).count()
        assert count > 100
    finally:
        session.close()

if __name__ == "__main__":
    pytest.main([__file__])
