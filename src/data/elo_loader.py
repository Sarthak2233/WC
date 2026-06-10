import pandas as pd
import logging
import os
from sqlalchemy.orm import Session
from src.data.base_loader import BaseLoader
from src.data.entity_resolver import get_iso3_code
from src.database import Elo

logger = logging.getLogger(__name__)

class EloLoader(BaseLoader):
    """
    Loads Elo ratings into the database.
    """
    def __init__(self, session_factory):
        self.session_factory = session_factory
        
    def extract(self) -> pd.DataFrame:
        path = os.path.join("data", "raw", "elo.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        
        # Add year (assuming the elo.csv is for 2026 context)
        df["year"] = 2026
        
        # Get ISO3 codes
        df["country_code"] = df["team"].apply(get_iso3_code)
        
        return df.dropna(subset=["country_code"])
        
    def load(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
            
        session: Session = self.session_factory()
        try:
            for _, row in df.iterrows():
                existing = session.query(Elo).filter_by(
                    country_code=row["country_code"],
                    year=row["year"]
                ).first()
                
                if not existing:
                    session.add(Elo(
                        country_code=row["country_code"],
                        year=row["year"],
                        elo=row["elo"]
                    ))
                else:
                    existing.elo = row["elo"]
            
            session.commit()
            logger.info("Successfully loaded Elo ratings.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading Elo data: {e}")
            raise
        finally:
            session.close()

    def run(self):
        df = self.extract()
        transformed = self.transform(df)
        self.load(transformed)
