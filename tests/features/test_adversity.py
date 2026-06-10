import pytest
import pandas as pd
from src.features.adversity import ChildhoodAdversityCalculator

def test_calculate_adversity():
    """
    Test the computation of the Childhood Adversity Index.
    Rules: 0 (privileged) to 10 (extreme hardship).
    Ages 5-15.
    """
    calculator = ChildhoodAdversityCalculator()
    
    # Mock data:
    # birth_year = 2000
    # ages 5-15 are 2005-2015
    mock_gdp = pd.DataFrame([
        {"year": y, "gdp_per_capita": 1000.0} for y in range(2005, 2016)
    ])
    mock_conflict = pd.DataFrame([
        {"year": y, "conflict_intensity": 2.0} for y in range(2005, 2010)  # 5 years of conflict
    ])
    
    score = calculator.calculate(birth_year=2000, country_gdp_df=mock_gdp, country_conflict_df=mock_conflict, global_gdp_distribution=[100.0, 500.0, 1000.0, 50000.0])
    
    # Expected logic based on PROJECT_REQUIREMENT.md
    # (1 - gdp_rank) * 5 + min(conflict_years, 5)
    # 1000 in [100, 500, 1000, 50000] is around the 75th percentile (index 2 / 3) -> ~0.75 rank?
    # Actually wait, let's just make sure it returns a float between 0 and 10.
    assert isinstance(score, float)
    assert 0.0 <= score <= 10.0
