import pandas as pd
from typing import Callable
from sqlalchemy.orm import Session

from src.database import WorldCup, Match, Player, Country, Culture, Conflict
from src.features.political import PoliticalFeatureCalculator
from src.features.psyche import NationalPsycheCalculator
from src.data.entity_resolver import get_iso3_code

class FeatureMatrixBuilder:
    """
    Assembles the 11 data layers into a Team-Tournament feature matrix for ML modeling.
    """
    
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory
        self.pol_calc = PoliticalFeatureCalculator()
        self.psyche_calc = NationalPsycheCalculator()
        
    def build(self, tournament_year: int) -> pd.DataFrame:
        """
        Builds the feature matrix for a specific tournament.
        One row per team per tournament.
        """
        session = self.session_factory()
        
        try:
            # 1. Identify participating teams from the tournament
            # We can find them from the players table for that year
            players = session.query(Player).filter_by(tournament_year=tournament_year).all()
            
            teams = set(p.country for p in players)
            
            rows = []
            
            wc = session.query(WorldCup).filter_by(year=tournament_year).first()
            host_str = wc.host if wc else ""
            hosts = [h.strip() for h in host_str.split("/")] if host_str else []
            
            for team in teams:
                iso3 = get_iso3_code(team)
                if not iso3:
                    # If we can't find iso3, use the team name as proxy
                    iso3 = team[:3].upper()
                    
                # A. Aggregate Player Features (Adversity)
                team_players = [p for p in players if p.country == team]
                adversity_scores = [p.adversity_score for p in team_players if p.adversity_score is not None]
                adv_mean = sum(adversity_scores)/len(adversity_scores) if adversity_scores else 5.0
                
                # B. Political & Economic Features (PPI)
                country = session.query(Country).filter_by(country_code=iso3, year=tournament_year).first()
                conflict = session.query(Conflict).filter_by(country_code=iso3, year=tournament_year).first()
                
                is_host = team in hosts
                ppi = self.pol_calc.calculate_ppi(country, conflict, is_host)
                
                # C. Culture & Psyche Features
                culture = session.query(Culture).filter_by(country_code=iso3).first()
                
                # We could pull choking_flag from historical matches, but default to 0.0 for now
                psyche_score = self.psyche_calc.calculate(culture, choking_flag=0.0)
                
                # D. Target Variable (if historical)
                # To be joined later or derived from Matches
                
                rows.append({
                    "tournament_year": tournament_year,
                    "team": team,
                    "country_code": iso3,
                    "is_host": is_host,
                    "adversity_mean": adv_mean,
                    "ppi": ppi,
                    "psyche_score": psyche_score
                    # To add: Elo, Legacy Burden, etc.
                })
                
            return pd.DataFrame(rows)
            
        finally:
            session.close()
