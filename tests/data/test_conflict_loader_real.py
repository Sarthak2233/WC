import pytest
from src.data.conflict_loader import ConflictLoader
from src.database import SessionLocal, init_db, Conflict

@pytest.fixture(scope="module")
def setup_db():
    init_db()
    yield SessionLocal

def test_conflict_loader_run(setup_db):
    loader = ConflictLoader(setup_db)
    
    # Run ETL (fetching UCDP ZIP)
    loader.run()
    
    session = setup_db()
    try:
        # Check for historical data (e.g., Vietnam or Ukraine)
        ukr_data = session.query(Conflict).filter_by(country_code="UKR", year=2022).first()
        if ukr_data:
            assert ukr_data.intensity > 0
            
        # Check count
        count = session.query(Conflict).count()
        assert count > 100
    finally:
        session.close()

if __name__ == "__main__":
    pytest.main([__file__])
