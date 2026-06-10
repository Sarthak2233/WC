import pytest
from src.features.psyche import NationalPsycheCalculator
from src.database import Culture

def test_calculate_psyche():
    """
    Test the computation of the Collective National Psyche Score.
    Based on PROJECT_REQUIREMENT.md:
    psyche_score = 0.4*UAI_norm + 0.3*(1-Trust_norm) + 0.3*Choking_history_flag
    """
    calculator = NationalPsycheCalculator()
    
    # Mock culture data
    culture = Culture(
        country_code="ENG",
        uai=80.0, # High uncertainty avoidance
        trust=0.2 # Low trust
    )
    
    # England historically chokes (based on the causal pattern library)
    choking_flag = 1.0 
    
    # Assuming UAI ranges 0-100, Trust 0-1
    # UAI norm = 80/100 = 0.8
    # Trust norm = 0.2/1 = 0.2
    # psyche = 0.4*0.8 + 0.3*(1-0.2) + 0.3*1.0
    # psyche = 0.32 + 0.24 + 0.3 = 0.86
    
    score = calculator.calculate(culture, choking_flag=choking_flag)
    
    assert isinstance(score, float)
    assert 0.85 < score < 0.87
    assert 0.0 <= score <= 1.0
