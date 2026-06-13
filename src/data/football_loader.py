import pandas as pd
import logging
import os
from typing import Dict, Any
import requests
from io import StringIO

from src.data.base_loader import BaseLoader
from src.data.entity_resolver import resolve_country_name
from src.data.scraper import SquadScraper

logger = logging.getLogger(__name__)

class FootballLoader(BaseLoader):
    """
    Loads football data including World Cup tournament metadata, matches, and players.
    Fetches historical data from GitHub and 2026 data from Wikipedia/local files.
    Saves cleaned data to data/processed/.
    """
    
    BASE_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/"
    
    def __init__(self):
        self.scraper = SquadScraper()
        self.processed_dir = os.path.join("data", "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def _fetch_csv(self, filename: str) -> pd.DataFrame:
        local_path = os.path.join("data", "raw", filename)
        if os.path.exists(local_path):
            logger.info(f"Loading {filename} from local cache.")
            return pd.read_csv(local_path)
            
        url = f"{self.BASE_URL}{filename}"
        logger.info(f"Fetching {filename} from remote.")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df.to_csv(local_path, index=False)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return pd.DataFrame()

    def extract(self) -> Dict[str, pd.DataFrame]:
        logger.info("Extracting historical football data from GitHub...")
        
        files = {
            "world_cups": "tournaments.csv",
            "matches": "matches.csv",
            "squads": "squads.csv",
            "players_meta": "players.csv",
            "goals": "goals.csv"
        }
        
        raw_data = {key: self._fetch_csv(filename) for key, filename in files.items()}
        
        logger.info("Scraping 2026 squads...")
        squads_2026 = self.scraper.scrape_2026_squads()
        
        if squads_2026.empty:
            local_squads = []
            raw_dir = os.path.join("data", "raw")
            for filename in os.listdir(raw_dir):
                if filename.startswith("fifawc26-squadlist-") and filename.endswith(".csv"):
                    team_name = filename.replace("fifawc26-squadlist-", "").replace(".csv", "")
                    df = pd.read_csv(os.path.join(raw_dir, filename))
                    df["team"] = team_name
                    local_squads.append(df)
            if local_squads:
                squads_2026 = pd.concat(local_squads, ignore_index=True)
        
        raw_data["squads_2026"] = squads_2026
        return raw_data
        
    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        logger.info("Transforming football data...")
        transformed = {}
        
        # 1. World Cups
        if "world_cups" in raw_data and not raw_data["world_cups"].empty:
            wc_df = raw_data["world_cups"].copy()
            wc_df["year"] = wc_df["tournament_id"].apply(lambda x: int(str(x).split("-")[1]))
            wc_df["host"] = wc_df["host_country"].apply(resolve_country_name)
            wc_df["winner"] = wc_df["winner"].apply(resolve_country_name)
            transformed["world_cups"] = wc_df[["year", "host", "winner"]]

        # 2. Matches
        if "matches" in raw_data and not raw_data["matches"].empty:
            match_df = raw_data["matches"].copy()
            match_df["tournament_year"] = match_df["tournament_id"].apply(lambda x: int(str(x).split("-")[1]))
            match_df["home_team"] = match_df["home_team_name"].apply(resolve_country_name)
            match_df["away_team"] = match_df["away_team_name"].apply(resolve_country_name)
            transformed["matches"] = match_df[["tournament_year", "stage_name", "home_team", "away_team", "home_team_score", "away_team_score"]]

        # 3. Players
        if "squads" in raw_data and not raw_data["squads"].empty:
            hist_players = raw_data["squads"].copy()
            hist_players["tournament_year"] = hist_players["tournament_id"].apply(lambda x: int(str(x).split("-")[1]))
            hist_players["country"] = hist_players["team_name"].apply(resolve_country_name)
            hist_players["name"] = hist_players["given_name"].fillna("") + " " + hist_players["family_name"].fillna("")
            hist_players["position"] = hist_players["position_name"]
            transformed["players"] = hist_players[["name", "country", "tournament_year", "position"]]
            
        return transformed

    def save_processed(self, transformed_data: Dict[str, pd.DataFrame]) -> None:
        """
        Saves transformed data to CSV in data/processed/.
        """
        for key, df in transformed_data.items():
            path = os.path.join(self.processed_dir, f"{key}.csv")
            df.to_csv(path, index=False)
            logger.info(f"Saved {key} to {path}")
