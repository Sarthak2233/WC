import requests
import pandas as pd
import logging
import os
import json
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from src.data.base_loader import BaseLoader
from src.database import Match, Player, SessionLocal
from src.fifa_database import PlayerRaw, FifaSessionLocal

logger = logging.getLogger(__name__)

class ApiFootballLoader(BaseLoader):
    """
    Loads football data from API-Football.
    """
    
    BASE_URL = "https://v3.football.api-sports.io"
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.api_key = os.getenv("API_FOOTBALL_KEY")
        self.raw_dir = os.path.join("data", "raw", "api_football")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.headers = {"x-apisports-key": self.api_key}

    def _fetch_json(self, endpoint: str, params: Dict[str, Any]) -> Any:
        url = f"{self.BASE_URL}{endpoint}"
        logger.info(f"Fetching from {url}")
        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        
        if response.status_code in [403, 429]:
            logger.warning(f"Received {response.status_code} for {url}. Skipping.")
            return None
            
        response.raise_for_status()
        return response.json()

    def extract(self) -> Dict[str, Any]:
        """
        Extracts 2026 World Cup data (Teams, Fixtures, Players) from API-Football.
        """
        if not self.api_key:
            logger.warning("API_FOOTBALL_KEY not set. Skipping API-Football extraction.")
            return {}

        logger.info("Extracting API-Football data (2026 World Cup)...")
        league_id = 1
        season = 2026
        
        cache_file = os.path.join(self.raw_dir, "football_api_data.json")
        if os.path.exists(cache_file):
            logger.info("Loading API-Football data from local cache.")
            with open(cache_file, 'r') as f:
                return json.load(f)

        teams = self._fetch_json(f"/teams", {"league": league_id, "season": season})
        fixtures = self._fetch_json(f"/fixtures", {"league": league_id, "season": season})
        players = self._fetch_json(f"/players", {"league": league_id, "season": season, "page": 1})
        
        raw_data = {"teams": teams, "fixtures": fixtures, "players": players}
        
        with open(cache_file, 'w') as f:
            json.dump(raw_data, f)
            
        return raw_data

    def transform(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        return raw_data

    def load(self, raw_data: Dict[str, Any]) -> None:
        """
        Loads API-Football data into wc_oracle and fifa_oracle databases.
        """
        if not raw_data:
            return

        # 1. Load Players into fifa_oracle.db (UPSERT)
        fifa_session = FifaSessionLocal()
        try:
            for item in raw_data.get("players", {}).get("response", []):
                p = item["player"]
                s = item["statistics"][0]
                existing = fifa_session.query(PlayerRaw).filter_by(full_name=p["name"]).first()
                if not existing:
                    fifa_session.add(PlayerRaw(
                        full_name=p["name"],
                        nationality=p["nationality"],
                        overall=s.get("games", {}).get("rating")
                    ))
            fifa_session.commit()
        except Exception as e:
            fifa_session.rollback()
            logger.error(f"Error loading fifa data: {e}")
        finally:
            fifa_session.close()

        # 2. Load Matches into wc_oracle.db
        wc_session = SessionLocal()
        try:
            for item in raw_data.get("fixtures", {}).get("response", []):
                f = item["fixture"]
                existing = wc_session.query(Match).filter_by(match_id=f["id"]).first()
                if not existing:
                    wc_session.add(Match(
                        match_id=f["id"],
                        home_team=item["teams"]["home"]["name"],
                        away_team=item["teams"]["away"]["name"]
                    ))
            wc_session.commit()
        except Exception as e:
            wc_session.rollback()
            logger.error(f"Error loading WC data: {e}")
        finally:
            wc_session.close()
            
        logger.info("Successfully loaded API-Football data (WC & FIFA databases).")
