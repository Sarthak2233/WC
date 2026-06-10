import numpy as np
from src.database import Country, Conflict

class PoliticalFeatureCalculator:
    """
    Computes Political Pressure Index (PPI) and Legacy Burden.
    """
    
    def calculate_ppi(self, country: Country, conflict: Conflict, is_host: bool) -> float:
        """
        PPI = (Political_Stability_index (inverted) * 0.5) + (Host_Flag * 0.3) + (Sanctions_Flag * 0.2)
        """
        if country is None:
            stability = 0.0
        else:
            stability = getattr(country, "political_stability", 0.0) or 0.0
            
        # Standardize WGI stability from [-2.5, 2.5] to [0, 1]
        stability_norm = (stability + 2.5) / 5.0
        stability_norm = np.clip(stability_norm, 0.0, 1.0)
        
        # Invert it so instability = higher pressure
        stability_inverted = 1.0 - stability_norm
        
        host_val = 1.0 if is_host else 0.0
        
        sanctions_val = 0.0
        if conflict and getattr(conflict, "sanctions_flag", False):
            sanctions_val = 1.0
            
        ppi = (stability_inverted * 0.5) + (host_val * 0.3) + (sanctions_val * 0.2)
        
        return float(np.clip(ppi, 0.0, 1.0))
        
    def calculate_legacy_burden(self, past_titles: int, years_since_last: int, fragmentation_flag: float) -> float:
        """
        burden = (Number_of_past_titles * 0.4) + (Years_since_last_title/50 * 0.3) + (National_identity_fragmentation_flag * 0.3)
        """
        # Use past_titles directly to match the exact formula:
        # (Number_of_past_titles * 0.4)
        titles_val = past_titles
        
        # Years since last: if never won, years_since_last is technically 0 for this formula (no legacy of winning)
        years_norm = 0.0
        if past_titles > 0:
            years_norm = np.clip(years_since_last / 50.0, 0.0, 1.0)
            
        frag_norm = np.clip(fragmentation_flag, 0.0, 1.0)
        
        burden = (titles_val * 0.4) + (years_norm * 0.3) + (frag_norm * 0.3)
        
        return float(np.clip(burden, 0.0, 1.0))
