from typing import Optional, List
from datetime import datetime
from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from .config import DATABASE_URL

class Base(DeclarativeBase):
    pass

# ==========================================
# 7 MASTER TABLES
# ==========================================

class WorldCup(Base):
    """Table 1: World Cups"""
    __tablename__ = "world_cups"
    
    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    host: Mapped[str] = mapped_column(String)
    winner: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    runner_up: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    format: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    num_teams: Mapped[int] = mapped_column(Integer)
    
    matches: Mapped[List["Match"]] = relationship(back_populates="tournament")

class Match(Base):
    """Table 2: Matches (Append-only ledger)"""
    __tablename__ = "matches"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tournament_year: Mapped[int] = mapped_column(ForeignKey("world_cups.year"))
    stage: Mapped[str] = mapped_column(String)
    date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    home_team: Mapped[str] = mapped_column(String)
    away_team: Mapped[str] = mapped_column(String)
    
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    extra_time: Mapped[bool] = mapped_column(Boolean, default=False)
    penalties: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Resolving "Knockout Scoring Rule" Open Question: 
    # For now, store the ultimate winner string directly to ensure correctness.
    actual_winner: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    
    tournament: Mapped["WorldCup"] = relationship(back_populates="matches")
    predictions: Mapped[List["Prediction"]] = relationship(back_populates="match")

class Player(Base):
    """Table 3: Players"""
    __tablename__ = "players"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    country: Mapped[str] = mapped_column(String)
    
    # Tournament context
    tournament_year: Mapped[int] = mapped_column(ForeignKey("world_cups.year"))
    
    # FIFA Attributes (to be populated from FIFA datasets)
    club: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    position: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    overall: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    potential: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    value_eur: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    wage_eur: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    pace_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shooting_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    passing_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    dribbling_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    defending_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    physicality_total: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Biographical / Contextual
    birth_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    caps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Synthetic feature (Childhood Adversity)
    adversity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

class Country(Base):
    """Table 4: Countries (Panel Data)"""
    __tablename__ = "countries"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    country_code: Mapped[str] = mapped_column(String, index=True) # ISO3
    year: Mapped[int] = mapped_column(Integer, index=True)
    
    # World Bank / Economy
    gdp_per_capita: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    political_stability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gini: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unemployment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

class Culture(Base):
    """Table 5: Culture"""
    __tablename__ = "culture"
    
    country_code: Mapped[str] = mapped_column(String, primary_key=True)
    
    # Hofstede Dimensions
    pdi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    idv: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mas: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    uai: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lto: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ivr: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # WVS
    trust: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    national_pride: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    happiness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

class Conflict(Base):
    """Table 6: Conflict"""
    __tablename__ = "conflict"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    country_code: Mapped[str] = mapped_column(String)
    year: Mapped[int] = mapped_column(Integer)
    
    conflict_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    intensity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sanctions_flag: Mapped[bool] = mapped_column(Boolean, default=False)

class Elo(Base):
    """Table 8: Elo Ratings"""
    __tablename__ = "elo_ratings"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    country_code: Mapped[str] = mapped_column(String, index=True)
    year: Mapped[int] = mapped_column(Integer, index=True)
    elo: Mapped[float] = mapped_column(Float)

class Narrative(Base):
    """Table 7: Narratives"""
    __tablename__ = "narratives"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    country_code: Mapped[str] = mapped_column(String)
    year: Mapped[int] = mapped_column(Integer)
    
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    identity_narrative: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    media_pressure: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


# ==========================================
# CONTEST & PREDICTION TABLES
# ==========================================

class Prediction(Base):
    """Contest Prediction"""
    __tablename__ = "predictions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    participant_name: Mapped[str] = mapped_column(String)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"))
    
    predicted_home_score: Mapped[int] = mapped_column(Integer)
    predicted_away_score: Mapped[int] = mapped_column(Integer)
    predicted_winner: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    locked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    points_awarded: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    match: Mapped["Match"] = relationship(back_populates="predictions")


# ==========================================
# SETUP
# ==========================================
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
