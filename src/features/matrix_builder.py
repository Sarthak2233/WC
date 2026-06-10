import pandas as pd
from typing import Callable
from sqlalchemy.orm import Session

from src.database import WorldCup, Match, Player, Country, Culture, Conflict, Elo
from src.features.political import PoliticalFeatureCalculator
from src.features.psyche import NationalPsycheCalculator
from src.features.adversity import ChildhoodAdversityCalculator
from src.data.entity_resolver import get_iso3_code

class FeatureMatrixBuilder:
    """
    Assembles the 11 data layers into a Team-Tournament feature matrix for ML modeling.
    """
    
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory
        self.pol_calc = PoliticalFeatureCalculator()
        self.psyche_calc = NationalPsycheCalculator()
        self.adv_calc = ChildhoodAdversityCalculator()
        
    def build(self, tournament_year: int) -> pd.DataFrame:
        """
        Builds the feature matrix for a specific tournament.
        One row per team per tournament.
        """
        session = self.session_factory()
        
        try:
            # 1. Identify participating teams from the tournament
            players = session.query(Player).filter_by(tournament_year=tournament_year).all()
            teams = set(p.country for p in players)
            
            rows = []
            
            wc = session.query(WorldCup).filter_by(year=tournament_year).first()
            host_str = wc.host if wc else ""
            hosts = [h.strip() for h in host_str.split("/")] if host_str else []
            
            # For Adversity, we need global GDP distribution for the relevant period
            # For simplicity, we'll fetch a sample from Country table
            gdp_sample = session.query(Country.gdp_per_capita).filter(
                Country.year <= tournament_year,
                Country.year >= tournament_year - 20
            ).all()
            global_gdp_dist = [g[0] for g in gdp_sample if g[0] is not None]
            
            for team in teams:
                iso3 = get_iso3_code(team)
                if not iso3:
                    iso3 = team[:3].upper()
                    
                # A. Aggregate Player Features (Adversity)
                team_players = [p for p in players if p.country == team]
                
                # In a full implementation, we'd fetch childhood GDP and conflict for each player.
                # Here we use the pre-calculated adversity_score if it exists, 
                # or calculate it on the fly if needed.
                adversity_scores = []
                for p in team_players:
                    if p.adversity_score is not None:
                        adversity_scores.append(p.adversity_score)
                    elif p.birth_year:
                        # Fetch country specific data for player's childhood
                        p_iso3 = get_iso3_code(p.country)
                        if p_iso3:
                            p_gdp = session.query(Country).filter_by(country_code=p_iso3).all()
                            p_conflict = session.query(Conflict).filter_by(country_code=p_iso3).all()
                            
                            p_gdp_df = pd.DataFrame([{"year": c.year, "gdp_per_capita": c.gdp_per_capita} for c in p_gdp])
                            p_conflict_df = pd.DataFrame([{"year": c.year, "conflict_intensity": c.intensity} for c in p_conflict])
                            
                            score = self.adv_calc.calculate(p.birth_year, p_gdp_df, p_conflict_df, global_gdp_dist)
                            adversity_scores.append(score)
                            
                adv_mean = sum(adversity_scores)/len(adversity_scores) if adversity_scores else 5.0
                
                # B. Political & Economic Features (PPI)
                country = session.query(Country).filter_by(country_code=iso3, year=tournament_year).first()
                # If exact year not found, try latest available before tournament
                if not country:
                    country = session.query(Country).filter(
                        Country.country_code == iso3,
                        Country.year < tournament_year
                    ).order_by(Country.year.desc()).first()
                    
                conflict = session.query(Conflict).filter_by(country_code=iso3, year=tournament_year).first()
                
                is_host = team in hosts
                ppi = self.pol_calc.calculate_ppi(country, conflict, is_host)
                
                # C. Legacy Burden
                # Count past titles
                past_wc = session.query(WorldCup).filter(WorldCup.year < tournament_year, WorldCup.winner == team).all()
                titles = len(past_wc)
                years_since = 0
                if titles > 0:
                    last_win_year = max(w.year for w in past_wc)
                    years_since = tournament_year - last_win_year
                
                # National identity fragmentation flag (placeholder: 0.2 default)
                legacy_burden = self.pol_calc.calculate_legacy_burden(titles, years_since, 0.2)
                
                # D. Culture & Psyche Features
                culture = session.query(Culture).filter_by(country_code=iso3).first()
                
                # E. Elo Rating
                elo_entry = session.query(Elo).filter_by(country_code=iso3, year=tournament_year).first()
                if not elo_entry:
                     elo_entry = session.query(Elo).filter(
                        Elo.country_code == iso3,
                        Elo.year < tournament_year
                    ).order_by(Elo.year.desc()).first()
                
                elo_val = elo_entry.elo if elo_entry else 1500.0
                
                # F. Psyche Score (can incorporate historical "choking" data if available)
                psyche_score = self.psyche_calc.calculate(culture, choking_flag=0.0)
                
                rows.append({
                    "tournament_year": tournament_year,
                    "team": team,
                    "country_code": iso3,
                    "is_host": is_host,
                    "elo": elo_val,
                    "adversity_mean": adv_mean,
                    "ppi": ppi,
                    "legacy_burden": legacy_burden,
                    "psyche_score": psyche_score
                })
                
            return pd.DataFrame(rows)
            
        finally:
            session.close()
