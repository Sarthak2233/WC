import pytest
from src.features.political import PoliticalFeatureCalculator
from src.database import Country, Conflict

def test_calculate_ppi():
    """
    PPI = (Political_Stability_index (inverted) * 0.5) + (Host_Flag * 0.3) + (Sanctions_Flag * 0.2)
    """
    calculator = PoliticalFeatureCalculator()
    
    country = Country(political_stability=-1.0) # Highly unstable -> high inverted score
    conflict = Conflict(sanctions_flag=True)
    
    # Stability -1.0 on a scale of roughly -2.5 to 2.5
    # Normalized ~ 0.2, inverted ~ 0.8
    # Host = 1
    # Sanctions = 1
    # Expected: (0.8 * 0.5) + (1.0 * 0.3) + (1.0 * 0.2) = 0.4 + 0.3 + 0.2 = 0.9
    
    ppi = calculator.calculate_ppi(country, conflict, is_host=True)
    
    assert isinstance(ppi, float)
    assert 0.0 <= ppi <= 1.0
    assert ppi > 0.5

def test_calculate_legacy_burden():
    """
    burden = (Number_of_past_titles * 0.4) + (Years_since_last_title/50 * 0.3) + (National_identity_fragmentation_flag * 0.3)
    """
    calculator = PoliticalFeatureCalculator()
    
    # E.g., England 1 title, 60 years ago (1966 to 2026)
    burden = calculator.calculate_legacy_burden(
        past_titles=1, 
        years_since_last=60, 
        fragmentation_flag=0.0
    )
    
    assert isinstance(burden, float)
    assert 0.0 <= burden <= 1.0
    # 1 title * 0.4 = 0.4
    # (min(60, 50)/50) * 0.3 = 0.3
    # = 0.7
    assert 0.69 < burden < 0.71
