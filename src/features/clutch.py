import numpy as np

class ClutchCalculator:
    """
    Calculates Player Clutch / Pressure Response Profile.
    """
    
    def calculate_player_clutch(self, ko_contrib_per_90: float, group_contrib_per_90: float) -> float:
        """
        pressure_factor = (ko_goal_contrib / group_goal_contrib)
        """
        # Handle zeros and small denominators
        if group_contrib_per_90 <= 0.001:
            if ko_contrib_per_90 <= 0.001:
                return 1.0  # Neutral if neither contributed
            return 5.0      # Max cap if they only contributed in knockouts (very clutch)
            
        factor = ko_contrib_per_90 / group_contrib_per_90
        
        # Cap the factor at 5.0 to prevent extreme outliers from small samples
        return float(np.clip(factor, 0.0, 5.0))
