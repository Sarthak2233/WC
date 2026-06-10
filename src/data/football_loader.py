import pandas as pd
import logging
import os
from typing import Dict, Any, List
from sqlalchemy.orm import Session
import requests
from io import StringIO

from src.data.base_loader import BaseLoader
from src.data.entity_resolver import resolve_country_name
from src.data.scraper import SquadScraper
from src.database import WorldCup, Match, Player

logger = logging.getLogger(__name__)

class FootballLoader(BaseLoader):
    """
    Loads football data including World Cup tournament metadata, matches, and players.
    Fetches historical data from GitHub and 2026 data from Wikipedia.
    """
    
    BASE_URL = "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/"
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.scraper = SquadScraper()
        
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
        """
        Extracts raw football data from GitHub (historical) and Wikipedia (2026).
        """
        logger.info("Extracting historical football data from GitHub...")
        
        # Explicitly fetch all required files
        files = {
            "world_cups": "tournaments.csv",
            "matches": "matches.csv",
            "squads": "squads.csv",
            "players_meta": "players.csv",
            "goals": "goals.csv"
        }
        
        raw_data = {}
        for key, filename in files.items():
            raw_data[key] = self._fetch_csv(filename)
        
        logger.info("Scraping 2026 squads from Wikipedia...")
        squads_2026 = self.scraper.scrape_2026_squads()
        
        # Fallback: check for local squadlist files
        if squads_2026.empty:
            logger.info("Wikipedia scraping returned no data. Falling back to local squadlist files...")
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
                logger.info(f"Loaded {len(squads_2026)} players from local squadlists.")
        
        raw_data["squads_2026"] = squads_2026
        
        return raw_data
        
    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transforms raw data to match DB schema.
        """
        logger.info("Transforming football data...")
        transformed = {}
        
        # 1. World Cups (Same)
        if "world_cups" in raw_data and not raw_data["world_cups"].empty:
            wc_df = raw_data["world_cups"].copy()
            logger.info(f"Transforming world_cups: {wc_df.shape}")
            wc_df["year"] = wc_df["tournament_id"].apply(lambda x: int(str(x).split("-")[1]))
            wc_df["host"] = wc_df["host_country"]
            wc_df["winner"] = wc_df["winner"]
            wc_df["num_teams"] = wc_df["count_teams"]
            wc_df["runner_up"] = None 
            for col in ["host", "winner"]:
                if col in wc_df.columns:
                    wc_df[col] = wc_df[col].apply(resolve_country_name)
            transformed["world_cups"] = wc_df[["year", "host", "winner", "runner_up", "num_teams"]]

        # 2. Matches (Same)
        if "matches" in raw_data and not raw_data["matches"].empty:
            match_df = raw_data["matches"].copy()
            logger.info(f"Transforming matches: {match_df.shape}")
            match_df["tournament_year"] = match_df["tournament_id"].apply(lambda x: int(str(x).split("-")[1]))
            match_df["stage"] = match_df["stage_name"]
            match_df["date"] = pd.to_datetime(match_df["match_date"], errors='coerce')
            match_df["home_team"] = match_df["home_team_name"].apply(resolve_country_name)
            match_df["away_team"] = match_df["away_team_name"].apply(resolve_country_name)
            match_df["home_score"] = match_df["home_team_score"]
            match_df["away_score"] = match_df["away_team_score"]
            match_df["extra_time"] = match_df["extra_time"]
            match_df["penalties"] = match_df["penalty_shootout"]
            match_df["actual_winner"] = None
            transformed["matches"] = match_df[["tournament_year", "stage", "date", "home_team", "away_team", "home_score", "away_score", "extra_time", "penalties", "actual_winner"]]

        # 3. Players & Attributes
        player_stats = pd.DataFrame()
        if "goals" in raw_data and not raw_data["goals"].empty:
            goals_df = raw_data["goals"].copy()
            # Own goals should not count for player
            total_goals = goals_df[goals_df["own_goal"] == 0].groupby("player_id").size().reset_index(name="goals")
            player_stats = total_goals
            
        p_meta = pd.DataFrame()
        if "players_meta" in raw_data and not raw_data["players_meta"].empty:
            p_meta = raw_data["players_meta"].copy()
            p_meta["birth_year"] = pd.to_datetime(p_meta["birth_date"], errors='coerce').dt.year
            # Merge goals into meta
            p_meta = pd.merge(p_meta, player_stats, on="player_id", how="left").fillna(0)
            
        # Join squad data with player_meta
        if "squads" in raw_data and not raw_data["squads"].empty:
            hist_players = raw_data["squads"].copy()
            logger.info(f"Transforming historical squads: {hist_players.shape}")
            hist_players["name"] = hist_players["given_name"].fillna("") + " " + hist_players["family_name"].fillna("")
            hist_players["name"] = hist_players["name"].str.strip()
            
            # Join with p_meta on player_id
            hist_players = pd.merge(hist_players, p_meta, on="player_id", how="left")
            
            hist_players = hist_players.rename(columns={
                "team_name": "country",
                "position_name": "position"
            })
            hist_players["tournament_year"] = hist_players["tournament_id"].apply(lambda x: int(str(x).split("-")[1]))
            hist_players["country"] = hist_players["country"].apply(resolve_country_name)
            
            transformed["players"] = hist_players[["name", "country", "tournament_year", "position", "birth_year", "goals"]]
            
        # 4. 2026 Squads
        if "squads_2026" in raw_data and not raw_data["squads_2026"].empty:
            squads_26 = raw_data["squads_2026"].copy()
            logger.info(f"Transforming 2026 squads: {squads_26.shape}. Columns: {list(squads_26.columns)}")
            
            # Normalize column names to uppercase for robustness
            squads_26.columns = [c.upper() for c in squads_26.columns]
            
            # Mapping based on observed local files and Wikipedia
            rename_map = {
                "PLAYER NAME": "name",
                "PLAYER_NAME": "name",
                "POS": "position",
                "TEAM": "country",
                "DOB": "birth_date",
                "NATIONALITY": "country"
            }
            # Handle potential duplicate columns after rename
            # Instead of rename, we'll selectively take columns
            
            def get_col(df, options):
                for opt in options:
                    if opt in df.columns:
                        return df[opt]
                return None

            p_26 = pd.DataFrame()
            p_26["name"] = get_col(squads_26, ["PLAYER NAME", "PLAYER_NAME"])
            p_26["position"] = get_col(squads_26, ["POS", "POSITION"])
            p_26["country"] = get_col(squads_26, ["TEAM", "NATIONALITY", "COUNTRY"])
            p_26["birth_date"] = get_col(squads_26, ["DOB", "BIRTH_DATE", "BIRTH DATE"])
            
            p_26["tournament_year"] = 2026
            
            # If name is missing but FIRST NAME(S) and LAST NAME(S) exist
            if p_26["name"] is None or p_26["name"].isna().all():
                 if "FIRST NAME(S)" in squads_26.columns and "LAST NAME(S)" in squads_26.columns:
                      p_26["name"] = squads_26["FIRST NAME(S)"].fillna("") + " " + squads_26["LAST NAME(S)"].fillna("")
            
            # Extract birth year
            p_26["birth_year"] = pd.to_datetime(p_26["birth_date"], errors='coerce').dt.year
                
            if "country" in p_26.columns and p_26["country"] is not None:
                p_26["country"] = p_26["country"].apply(lambda x: resolve_country_name(str(x)) if pd.notna(x) else "")
            
            # Clean up: only keep columns that exist in the Player model
            p_26 = p_26[["name", "country", "tournament_year", "position", "birth_year"]]
            p_26 = p_26.dropna(subset=["name", "country"])
            
            if "players" in transformed:
                logger.info("Concatenating 2026 players with historical players")
                transformed["players"] = pd.concat([transformed["players"], p_26], ignore_index=True)
            else:
                transformed["players"] = p_26
        
        if "players" in transformed:
             logger.info(f"Total players transformed: {transformed['players'].shape}")
            
        return transformed

    def load(self, transformed_data: Dict[str, pd.DataFrame]) -> None:
        """
        Loads transformed data into the Master Tables with UPSERT logic.
        """
        session: Session = self.session_factory()
        try:
            # Load World Cups
            if "world_cups" in transformed_data:
                for _, row in transformed_data["world_cups"].iterrows():
                    existing = session.query(WorldCup).filter_by(year=row["year"]).first()
                    if not existing:
                        wc = WorldCup(**row.to_dict())
                        session.add(wc)
                session.flush()

            # Load Matches
            if "matches" in transformed_data:
                for idx, row in transformed_data["matches"].iterrows():
                    try:
                        # Simplified idempotency: check tournament, stage, teams
                        existing = session.query(Match).filter_by(
                            tournament_year=row["tournament_year"],
                            home_team=row["home_team"],
                            away_team=row["away_team"]
                        ).first()
                        if not existing:
                            match_data = row.to_dict()
                            # Remove NaT from date safely
                            val_date = match_data.get("date")
                            if pd.isna(val_date) if not hasattr(val_date, "__len__") else True:
                                match_data["date"] = None
                            session.add(Match(**match_data))
                    except Exception as e:
                        logger.error(f"Error processing match at row {idx}: {e}")
                        raise
                session.flush()

            # Load Players (UPSERT)
            if "players" in transformed_data:
                for idx, row in transformed_data["players"].iterrows():
                    try:
                        player_data = row.to_dict()

                        # Ensure name and country are strings, not series
                        p_name = str(player_data["name"])
                        p_country = str(player_data["country"])
                        p_year = int(player_data["tournament_year"])

                        existing = session.query(Player).filter_by(
                            name=p_name,
                            country=p_country,
                            tournament_year=p_year
                        ).first()

                        if not existing:
                            # Clean up player_data to ensure no series/lists
                            clean_data = {}
                            for k, v in player_data.items():
                                if hasattr(v, "__len__") and not isinstance(v, str):
                                    clean_data[k] = None # or some default
                                elif pd.isna(v):
                                    clean_data[k] = None
                                else:
                                    clean_data[k] = v
                            session.add(Player(**clean_data))
                        else:
                            # Update existing
                            for k, v in player_data.items():
                                if not (hasattr(v, "__len__") and not isinstance(v, str)) and pd.notna(v):
                                    setattr(existing, k, v)
                    except Exception as e:
                        logger.error(f"Error processing player at row {idx}: {e}")
                        raise
            session.commit()
            logger.info("Successfully loaded football data (UPSERT).")
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading football data: {e}")
            raise
        finally:
            session.close()
