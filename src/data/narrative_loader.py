import pandas as pd
import requests
import logging
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from src.data.base_loader import BaseLoader
from src.data.entity_resolver import resolve_fips_to_iso3
from src.database import Narrative

logger = logging.getLogger(__name__)

import time

class NarrativeLoader(BaseLoader):
    """
    Loads narrative data (Layer 11) from GDELT DOC API 2.0.
    Fetches sentiment (tone) timelines for countries.
    """
    
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        
    def _fetch_tone_timeline(self, fips_code: str) -> List[Dict[str, Any]]:
        params = {
            "query": f"sourcecountry:{fips_code}",
            "mode": "timelinetone",
            "format": "json",
            "timespan": "1y" # Fetch last year for recent narrative
        }
        for attempt in range(3):
            try:
                # Add delay (exponential backoff)
                time.sleep(10 * (attempt + 1))
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                if response.status_code == 429:
                    logger.warning(f"Rate limited (429) for {fips_code}. Retrying...")
                    continue
                response.raise_for_status()
                data = response.json()
                return data.get("timeline", [{}])[0].get("data", [])
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {fips_code}: {e}")
        return []

    def extract(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extracts narrative data for key World Cup countries.
        In Layer 11 MVP, we focus on a subset or all common FIPS.
        """
        logger.info("Extracting narrative data from GDELT...")
        # Subset of common World Cup FIPS for MVP
        key_fips = ["US", "UK", "GM", "BR", "AR", "FR", "SP", "IT", "JA", "KS"]
        
        results = {}
        for fips in key_fips:
            results[fips] = self._fetch_tone_timeline(fips)
            
        return results
        
    def transform(self, raw_data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Transforms GDELT timelines into a unified country-year-sentiment format.
        """
        rows = []
        for fips, timeline in raw_data.items():
            iso3 = resolve_fips_to_iso3(fips)
            if not iso3: continue
            
            for entry in timeline:
                # date format: 20231001T000000Z
                date_str = entry.get("date")
                if date_str:
                    year = int(date_str[:4])
                    rows.append({
                        "country_code": iso3,
                        "year": year,
                        "sentiment_score": float(entry.get("value", 0))
                    })
        
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)
        # Average sentiment by country-year
        agg = df.groupby(["country_code", "year"])["sentiment_score"].mean().reset_index()
        return agg
        
    def load(self, df: pd.DataFrame) -> None:
        """
        Loads narrative data into the Narrative table.
        """
        if df.empty:
            return
            
        session: Session = self.session_factory()
        try:
            for _, row in df.iterrows():
                existing = session.query(Narrative).filter_by(
                    country_code=row["country_code"],
                    year=row["year"]
                ).first()
                
                if not existing:
                    session.add(Narrative(
                        country_code=row["country_code"],
                        year=row["year"],
                        sentiment_score=row["sentiment_score"]
                    ))
                else:
                    existing.sentiment_score = row["sentiment_score"]
                    
            session.commit()
            logger.info("Successfully loaded Layer 11: Narrative data.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading narrative data: {e}")
            raise
        finally:
            session.close()
