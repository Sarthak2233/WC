import pytest
import os
import pandas as pd
from src.data.political_loader import PoliticalLoader

def test_political_loader_run():
    loader = PoliticalLoader()
    
    # Run ETL
    loader.run()
    
    output_path = os.path.join("data", "processed", "political_economic.csv")
    assert os.path.exists(output_path)
    
    # Check data
    df = pd.read_csv(output_path)
    assert not df.empty
    assert "country_code" in df.columns
    assert "political_stability" in df.columns
    assert "gdp_per_capita" in df.columns
    
    # Check if we have data for a major country
    usa_data = df[(df["country_code"] == "USA") & (df["year"] == 2021)]
    assert not usa_data.empty
    assert usa_data["gdp_per_capita"].iloc[0] > 0

if __name__ == "__main__":
    pytest.main([__file__])
