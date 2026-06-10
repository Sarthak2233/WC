from src.database import Culture
import numpy as np

class NationalPsycheCalculator:
    """
    Computes the Collective National Psyche Score.
    A composite metric of how a country's population and team typically react under pressure.
    """
    
    def calculate(self, culture_data: Culture, choking_flag: float = 0.0) -> float:
        """
        Calculates the Psyche Score.
        Formula from PROJECT_REQUIREMENT.md:
        psyche_score = 0.4*UAI_norm + 0.3*(1-Trust_norm) + 0.3*Choking_history_flag
        """
        if culture_data is None:
            return 0.5 # Neutral
            
        uai = getattr(culture_data, "uai", 50.0) or 50.0
        trust = getattr(culture_data, "trust", 0.5) or 0.5
        
        # Normalize UAI (typically 0-100 in Hofstede)
        uai_norm = uai / 100.0
        uai_norm = np.clip(uai_norm, 0.0, 1.0)
        
        # Normalize Trust (typically 0-1 in WVS, if not clip it)
        trust_norm = np.clip(trust, 0.0, 1.0)
        
        # Choking flag is expected to be 0.0 or 1.0
        choke_norm = np.clip(choking_flag, 0.0, 1.0)
        
        score = (0.4 * uai_norm) + (0.3 * (1.0 - trust_norm)) + (0.3 * choke_norm)
        
        return float(np.clip(score, 0.0, 1.0))
