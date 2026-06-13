import requests
import pandas as pd
import logging
import os
from typing import Dict, Any, List

from src.data.base_loader import BaseLoader

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
    
    def __init__(self, session_factory=None):
        self.processed_dir = os.path.join("data", "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
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
            is_knockout = m.get("competition_stage", {}).get("name") != "Regular Season"
            
            data.append({
                "match_id": m["match_id"],
                "year": m["tournament_year"],
                "home_team": m["home_team"]["home_team_name"],
                "away_team": m["away_team"]["away_team_name"],
                "is_knockout": is_knockout,
                "penalties": m.get("home_score") == m.get("away_score") and is_knockout 
            })
        return pd.DataFrame(data)
        
    def save_processed(self, df: pd.DataFrame) -> None:
        """
        Saves transformed data to CSV in data/processed/.
        """
        path = os.path.join(self.processed_dir, "performance_metadata.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved performance metadata to {path}")
