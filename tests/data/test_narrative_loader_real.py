import pytest
import os
import pandas as pd
from src.data.narrative_loader import NarrativeLoader

def test_narrative_loader_run():
    loader = NarrativeLoader()
    
    # Run ETL (fetching GDELT)
    loader.run()
    
    output_path = os.path.join("data", "processed", "narrative_data.csv")
    assert os.path.exists(output_path)
    
    # Check data
    df = pd.read_csv(output_path)
    assert not df.empty
    assert "country_code" in df.columns
    assert "year" in df.columns
    assert "sentiment_score" in df.columns
    
    # Check if we have data for a major country
    usa_data = df[df["country_code"] == "USA"]
    assert not usa_data.empty

if __name__ == "__main__":
    pytest.main([__file__])
