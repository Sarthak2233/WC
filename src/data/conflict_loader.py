import requests
import pandas as pd
import logging
import os
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from src.data.base_loader import BaseLoader
from src.database import Conflict

logger = logging.getLogger(__name__)

class ConflictLoader(BaseLoader):
    """
    Loads conflict data (Layer 7) from ACLED (API) and UCDP (CSV mirror).
    """
    
    # UCDP Mirror on GitHub or official direct link (using mirror for stability in MVP)
    UCDP_URL = "https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-251-csv.zip"
    ACLED_URL = "https://api.acleddata.com/acled/read"
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.acled_email = os.getenv("ACLED_EMAIL")
        self.acled_password = os.getenv("ACLED_PASSWORD")
        self.raw_dir = os.path.join("data", "raw", "conflict")
        os.makedirs(self.raw_dir, exist_ok=True)

    def _get_access_token(self) -> str:
        token_url = "https://acleddata.com/oauth/token"
        data = {
            'username': self.acled_email,
            'password': self.acled_password,
            'grant_type': "password",
            'client_id': "acled",
            'scope': "authenticated"
        }
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        return response.json()['access_token']
        
    def _fetch_acled(self, country: str = None, year: int = None) -> pd.DataFrame:
        if not self.acled_email or not self.acled_password:
            logger.warning("ACLED credentials not found. Skipping ACLED extraction.")
            return pd.DataFrame()
            
        token = self._get_access_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        params = {
            "terms_of_use": "true",
            "limit": 5000
        }
        if country: params["country"] = country
        if year: params["year"] = year
        
        try:
            response = requests.get(self.ACLED_URL, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                return pd.DataFrame(data["data"])
            else:
                logger.error(f"ACLED API error: {data.get('error')}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"ACLED fetch failed: {e}")
            return pd.DataFrame()

    def _fetch_ucdp(self) -> pd.DataFrame:
        local_path = os.path.join("data", "raw", "ucdp_prio_acd_251.csv.zip")
        if os.path.exists(local_path):
            logger.info("Loading UCDP data from local cache.")
            return pd.read_csv(local_path, compression='zip')
            
        try:
            logger.info("Fetching UCDP data from remote.")
            response = requests.get(self.UCDP_URL, timeout=30)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return pd.read_csv(local_path, compression='zip')
        except Exception as e:
            logger.error(f"UCDP fetch failed: {e}")
            return pd.DataFrame()

    def extract(self) -> Dict[str, pd.DataFrame]:
        """
        Extracts conflict data from ACLED (recent) and UCDP (historical).
        """
        logger.info("Extracting conflict data...")
        # For MVP, we extract UCDP for broad historical coverage
        ucdp_df = self._fetch_ucdp()
        
        # ACLED for 2024-2026 (requires keys)
        acled_df = self._fetch_acled()
        
        return {
            "ucdp": ucdp_df,
            "acled": acled_df
        }
        
    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Transforms conflict data into a unified country-year intensity format.
        """
        results = []
        
        # 1. Transform UCDP (1946-2024)
        if not raw_data["ucdp"].empty:
            ucdp = raw_data["ucdp"].copy()
            # Columns: conflict_id, location, side_a, side_b, year, intensity_level
            # intensity_level: 1: minor (25-999 deaths), 2: war (>1000 deaths)
            for _, row in ucdp.iterrows():
                # UCDP 'location' can have multiple countries separated by ','
                locations = str(row["location"]).split(",")
                for loc in locations:
                    results.append({
                        "country": loc.strip(),
                        "year": int(row["year"]),
                        "conflict_type": "UCDP Armed Conflict",
                        "intensity": float(row["intensity_level"])
                    })
        
        # 2. Transform ACLED (if available)
        if not raw_data["acled"].empty:
            acled = raw_data["acled"].copy()
            # Aggregate ACLED into country-year fatalities
            # Columns: country, year, fatalities
            acled["fatalities"] = pd.to_numeric(acled["fatalities"], errors='coerce').fillna(0)
            agg = acled.groupby(["country", "year"])["fatalities"].sum().reset_index()
            for _, row in agg.iterrows():
                results.append({
                    "country": row["country"],
                    "year": int(row["year"]),
                    "conflict_type": "ACLED Event",
                    "intensity": float(row["fatalities"]) / 1000.0 # Normalize roughly to UCDP scale
                })
        
        return pd.DataFrame(results)
        
    def load(self, df: pd.DataFrame) -> None:
        """
        Loads transformed conflict data into the Conflict table.
        """
        if df.empty:
            return
            
        from src.data.entity_resolver import get_iso3_code
        session: Session = self.session_factory()
        try:
            # Map country names to ISO3
            df["country_code"] = df["country"].apply(get_iso3_code)
            df = df.dropna(subset=["country_code"])
            
            # Aggregate duplicates by country/year (max intensity)
            df = df.groupby(["country_code", "year"]).agg({
                "intensity": "max",
                "conflict_type": "first"
            }).reset_index()
            
            for _, row in df.iterrows():
                existing = session.query(Conflict).filter_by(
                    country_code=row["country_code"],
                    year=row["year"]
                ).first()
                
                if not existing:
                    session.add(Conflict(
                        country_code=row["country_code"],
                        year=row["year"],
                        conflict_type=row["conflict_type"],
                        intensity=row["intensity"]
                    ))
                else:
                    existing.intensity = max(existing.intensity or 0, row["intensity"])
                    
            session.commit()
            logger.info("Successfully loaded Layer 7: Conflict data.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading conflict data: {e}")
            raise
        finally:
            session.close()
