from src.database import Prediction
from src.config import STAGE_SCORING

class ContestScorer:
    """
    Implements the scoring system defined in RULE-AND-SYSTEM.md.
    """
    
    def calculate_score(self, prediction: Prediction, actual_home: int, actual_away: int, stage: str) -> int:
        """
        Calculates the score for a single prediction based on the actual result and the tournament stage.
        """
        if stage not in STAGE_SCORING:
            # Fallback to group stage scoring if unknown
            stage = "Group Stage"
            
        scoring_rules = STAGE_SCORING[stage]
        
        pred_home = prediction.predicted_home_score
        pred_away = prediction.predicted_away_score
        
        # 1. Check exact match
        if pred_home == actual_home and pred_away == actual_away:
            return scoring_rules["exact"]
            
        # 2. Check correct result (Win, Loss, Draw)
        pred_diff = pred_home - pred_away
        actual_diff = actual_home - actual_away
        
        # Both positive (home win), both negative (away win), or both zero (draw)
        if (pred_diff > 0 and actual_diff > 0) or \
           (pred_diff < 0 and actual_diff < 0) or \
           (pred_diff == 0 and actual_diff == 0):
            return scoring_rules["result"]
            
        # 3. Incorrect result
        return 0
