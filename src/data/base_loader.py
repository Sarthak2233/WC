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
    def transform(self, raw_data: Any) -> Any:
        """Transform raw data into the schema required."""
        pass
        
    @abstractmethod
    def save_processed(self, transformed_data: Any) -> None:
        """Saves the transformed data to CSV."""
        pass
        
    def run(self) -> None:
        """Execute the full ETL pipeline."""
        raw_data = self.extract()
        if raw_data is not None:
            transformed_data = self.transform(raw_data)
            if transformed_data is not None:
                self.save_processed(transformed_data)
