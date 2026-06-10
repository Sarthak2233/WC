import logging
import os
from sqlalchemy import inspect
from src.database import SessionLocal, init_db, WorldCup, Match, Player, Country, Culture, Conflict, Narrative
from src.fifa_database import init_fifa_db, FifaSessionLocal, PlayerRaw
from src.data.football_loader import FootballLoader
from src.data.performance_loader import PerformanceLoader
from src.data.squad_loader import SquadLoader
from src.data.political_loader import PoliticalLoader
from src.data.conflict_loader import ConflictLoader
from src.data.culture_loader import CultureLoader
from src.data.narrative_loader import NarrativeLoader
from src.data.psyche_loader import PsycheLoader
from src.data.fifa_loader import FifaLoader
from src.data.api_football_loader import ApiFootballLoader

logger = logging.getLogger(__name__)

def check_is_populated(session, model, columns):
    """Checks if a representative subset of data is populated for given columns."""
    for col in columns:
        if session.query(model).filter(getattr(model, col).isnot(None)).first() is None:
            return False
    return True

def run_all_layers():
    """
    Orchestrates the 11-layer data gathering process with granular idempotency checks.
    """
    logger.info("Starting 11-Layer Data Gathering Process...")
    
    init_db()
    init_fifa_db()
    
    # Mapping loaders to their primary database models and session factories
    tasks = [
        (FootballLoader(SessionLocal), WorldCup, SessionLocal, []),
        (PoliticalLoader(SessionLocal), Country, SessionLocal, ["political_stability"]),
        (CultureLoader(SessionLocal), Culture, SessionLocal, ["pdi"]),
        (ConflictLoader(SessionLocal), Conflict, SessionLocal, ["intensity"]),
        (NarrativeLoader(SessionLocal), Narrative, SessionLocal, ["sentiment_score"]),
        (PerformanceLoader(SessionLocal), Match, SessionLocal, []),
        (FifaLoader(FifaSessionLocal), PlayerRaw, FifaSessionLocal, ["overall", "club"]),
        (ApiFootballLoader(FifaSessionLocal), PlayerRaw, FifaSessionLocal, ["overall"]), 
        (SquadLoader(SessionLocal), Player, SessionLocal, ["club"]),
        (PsycheLoader(SessionLocal), Player, SessionLocal, ["adversity_score"]),
    ]
    
    for loader, model, session_factory, critical_cols in tasks:
        layer_name = loader.__class__.__name__
        
        # Idempotency Check
        session = session_factory()
        
        skip = False
        if layer_name == "ApiFootballLoader":
            # Always run for API integration to get latest 2026 data
            skip = False
        elif layer_name == "PsycheLoader":
            # Force run
            skip = False
        else:
            is_empty = session.query(model).first() is None
            if not is_empty:
                # Granular check: only skip if critical columns are populated
                if critical_cols and check_is_populated(session, model, critical_cols):
                    skip = True
                elif not critical_cols:
                    # Default: skip if table is not empty
                    skip = True
        
        session.close()
        
        if skip:
            logger.info(f"Skipping {layer_name} (data already present).")
            continue
            
        try:
            logger.info(f"Running {layer_name}...")
            loader.run()
            logger.info(f"Finished {layer_name}.")
        except Exception as e:
            logger.error(f"Failed to run {layer_name}: {e}")
            continue

    logger.info("11-Layer Data Gathering Complete.")

if __name__ == "__main__":
    run_all_layers()
