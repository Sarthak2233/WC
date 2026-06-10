from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class BaseLoader(ABC):
    """
    Abstract base class for all ETL loaders in the World Cup Oracle.
    Enforces idempotent extract-transform-load patterns per DSA_POLICY.md.
    """
    
    @abstractmethod
    def extract(self) -> Any:
        """Extract data from source (API, Web, CSV)."""
        pass
        
    @abstractmethod
    def transform(self, raw_data: Any) -> pd.DataFrame:
        """Transform raw data into the schema required by the Master Tables."""
        pass
        
    @abstractmethod
    def load(self, df: pd.DataFrame) -> None:
        """Load the transformed data into the Master Tables."""
        pass
        
    def run(self) -> None:
        """Execute the full ETL pipeline."""
        raw_data = self.extract()
        if raw_data is not None:
            transformed_data = self.transform(raw_data)
            # Fix: Avoid ambiguous truth value check for DataFrames
            if transformed_data is not None:
                if isinstance(transformed_data, pd.DataFrame):
                    if not transformed_data.empty:
                        self.load(transformed_data)
                elif isinstance(transformed_data, dict):
                    if transformed_data:
                        self.load(transformed_data)
                else:
                    # Fallback for other types
                    self.load(transformed_data)
