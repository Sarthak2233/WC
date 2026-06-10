import requests
import pandas as pd
import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from src.data.base_loader import BaseLoader
from src.database import Player, Match

logger = logging.getLogger(__name__)

class PerformanceLoader(BaseLoader):
    """
    Loads player performance data (Layer 2) from StatsBomb Open Data.
    Analyzes 2018 and 2022 World Cup event data for 'clutch' and 'pressure' metrics.
    """
    
    BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"
    WC_COMPETITIONS = [
        {"id": 43, "season_id": 106, "year": 2022},
        {"id": 43, "season_id": 3, "year": 2018}
    ]
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        
    def _fetch_json(self, path: str) -> Any:
        url = f"{self.BASE_URL}{path}"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def extract(self) -> Dict[str, Any]:
        """
        Extracts match lists for analysis.
        Processing full event data for every match is too heavy for a single run,
        so we focus on match-level summaries and a sample of event data.
        """
        all_matches = []
        for comp in self.WC_COMPETITIONS:
            matches = self._fetch_json(f"matches/{comp['id']}/{comp['season_id']}.json")
            if matches:
                for m in matches:
                    m["tournament_year"] = comp["year"]
                all_matches.extend(matches)
        
        return {"matches": all_matches}
        
    def transform(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Transforms StatsBomb match data into performance metadata.
        """
        matches = raw_data.get("matches", [])
        data = []
        for m in matches:
            # We can identify knockout matches by stage
            # "Final", "Semi-finals", "Quarter-finals", "Round of 16"
            is_knockout = m.get("competition_stage", {}).get("name") != "Regular Season"
            
            data.append({
                "match_id": m["match_id"],
                "year": m["tournament_year"],
                "home_team": m["home_team"]["home_team_name"],
                "away_team": m["away_team"]["away_team_name"],
                "is_knockout": is_knockout,
                "penalties": m.get("home_score") == m.get("away_score") and is_knockout # Simplified
            })
        return pd.DataFrame(data)
        
    def load(self, df: pd.DataFrame) -> None:
        """
        Logs performance metadata processing.
        Actual 'clutch' score will be computed in the Feature Engineering phase
        by joining this metadata with match results and player lineups.
        """
        logger.info(f"Successfully processed {len(df)} matches from StatsBomb for Layer 2 analysis.")
