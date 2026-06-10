import pytest
from src.data.narrative_loader import NarrativeLoader
from src.database import SessionLocal, init_db, Narrative

@pytest.fixture(scope="module")
def setup_db():
    init_db()
    yield SessionLocal

def test_narrative_loader_run(setup_db):
    loader = NarrativeLoader(setup_db)
    
    # Run ETL (fetching GDELT)
    loader.run()
    
    session = setup_db()
    try:
        # Check for recent data (USA or GBR)
        usa_data = session.query(Narrative).filter_by(country_code="USA").first()
        if usa_data:
            assert usa_data.sentiment_score is not None
            
        # Check count
        count = session.query(Narrative).count()
        assert count > 0
    finally:
        session.close()

if __name__ == "__main__":
    pytest.main([__file__])
