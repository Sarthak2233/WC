import pytest
from src.data.culture_loader import CultureLoader
from src.database import SessionLocal, init_db, Culture

@pytest.fixture(scope="module")
def setup_db():
    init_db()
    yield SessionLocal

def test_culture_loader_run(setup_db):
    loader = CultureLoader(setup_db)
    
    # Run ETL
    loader.run()
    
    session = setup_db()
    try:
        # Check if we have data for USA
        usa_data = session.query(Culture).filter_by(country_code="USA").first()
        assert usa_data is not None
        assert usa_data.idv > 80 # USA is high on individualism
        assert usa_data.happiness_score > 0
        
        # Check count
        count = session.query(Culture).count()
        assert count > 50
    finally:
        session.close()

if __name__ == "__main__":
    pytest.main([__file__])
