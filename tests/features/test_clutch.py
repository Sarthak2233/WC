import pytest
from src.features.clutch import ClutchCalculator

def test_calculate_clutch_factor():
    """
    pressure_factor = (ko_goal_contrib / group_goal_contrib)
    """
    calculator = ClutchCalculator()
    
    # Example 1: Player performs better in knockouts
    ko_contrib = 1.0 # 1 goal/assist per 90
    group_contrib = 0.5 # 0.5 goal/assist per 90
    factor = calculator.calculate_player_clutch(ko_contrib, group_contrib)
    assert factor == 2.0
    
    # Example 2: Player performs worse in knockouts
    factor2 = calculator.calculate_player_clutch(0.2, 0.8)
    assert factor2 == 0.25
    
    # Example 3: Zero group contrib (avoid division by zero)
    factor3 = calculator.calculate_player_clutch(1.0, 0.0)
    # If group is 0 and KO is > 0, clutch factor should be high.
    # The implementation should handle this gracefully (e.g. return a max value or ko_contrib / small_epsilon)
    assert factor3 > 1.0
    
    # Example 4: Both zero
    factor4 = calculator.calculate_player_clutch(0.0, 0.0)
    assert factor4 == 1.0 # Neutral
