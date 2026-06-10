import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database import Base, WorldCup, Match, Player, Country, Culture, Conflict, Narrative, Prediction

@pytest.fixture(scope="module")
def test_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(test_engine):
    """Provide a transactional scope around a series of operations."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()

def test_world_cup_creation(db_session):
    """Test inserting a record into the WorldCups table."""
    wc = WorldCup(
        year=2026,
        host="USA/Canada/Mexico",
        format="Groups of 4, Round of 32",
        num_teams=48
    )
    db_session.add(wc)
    db_session.commit()
    
    retrieved = db_session.query(WorldCup).filter_by(year=2026).first()
    assert retrieved is not None
    assert retrieved.host == "USA/Canada/Mexico"
    assert retrieved.num_teams == 48

def test_match_creation(db_session):
    """Test inserting a Match and linking it to a WorldCup."""
    wc = WorldCup(year=2022, host="Qatar", num_teams=32)
    db_session.add(wc)
    
    match = Match(
        tournament_year=2022,
        stage="Final",
        home_team="Argentina",
        away_team="France",
        home_score=3,
        away_score=3,
        extra_time=True,
        penalties=True,
        actual_winner="Argentina"
    )
    db_session.add(match)
    db_session.commit()
    
    retrieved_match = db_session.query(Match).filter_by(stage="Final", tournament_year=2022).first()
    assert retrieved_match is not None
    assert retrieved_match.actual_winner == "Argentina"
    assert retrieved_match.tournament.host == "Qatar"

def test_player_creation(db_session):
    """Test inserting a Player and linking to tournament."""
    wc = WorldCup(year=2030, host="Spain/Portugal/Morocco", num_teams=48)
    db_session.add(wc)
    
    player = Player(
        name="Lionel Messi",
        country="Argentina",
        birth_year=1987,
        position="Forward",
        tournament_year=2030,
        caps=180,
        goals=106,
        adversity_score=6.5
    )
    db_session.add(player)
    db_session.commit()
    
    retrieved = db_session.query(Player).filter_by(name="Lionel Messi", tournament_year=2030).first()
    assert retrieved is not None
    assert retrieved.adversity_score == 6.5
