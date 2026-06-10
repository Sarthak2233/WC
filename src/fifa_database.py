from sqlalchemy import String, Integer, Float, create_engine, Column, ForeignKey
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from .config import FIFA_DATABASE_URL

class Base(DeclarativeBase):
    pass

class PlayerRaw(Base):
    """Expanded Raw FIFA Player Attributes"""
    __tablename__ = "players_raw"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String, nullable=False)
    nationality = Column(String, nullable=False)
    club = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    overall = Column(Integer, nullable=True)
    potential = Column(Integer, nullable=True)
    best_position = Column(String, nullable=True)
    
    # Expanded attributes
    height_cm = Column(Integer, nullable=True)
    weight_kg = Column(Integer, nullable=True)
    value_eur = Column(Integer, nullable=True)
    wage_eur = Column(Integer, nullable=True)
    pace_total = Column(Integer, nullable=True)
    shooting_total = Column(Integer, nullable=True)
    passing_total = Column(Integer, nullable=True)
    dribbling_total = Column(Integer, nullable=True)
    defending_total = Column(Integer, nullable=True)
    physicality_total = Column(Integer, nullable=True)
    crossing = Column(Integer, nullable=True)
    finishing = Column(Integer, nullable=True)
    short_passing = Column(Integer, nullable=True)
    dribbling = Column(Integer, nullable=True)
    stamina = Column(Integer, nullable=True)
    strength = Column(Integer, nullable=True)
    vision = Column(Integer, nullable=True)
    penalties = Column(Integer, nullable=True)
    composure = Column(Integer, nullable=True)

# Setup
fifa_engine = create_engine(FIFA_DATABASE_URL, echo=False)
FifaSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fifa_engine)

def init_fifa_db() -> None:
    """Create raw FIFA tables."""
    Base.metadata.create_all(bind=fifa_engine)
