import pytest
import pandas as pd
import os
from src.features.csv_oracle import CSVFeatureOracle

@pytest.fixture
def oracle():
    return CSVFeatureOracle()

def test_oracle_load_layers(oracle):
    """
    Test that the oracle loads the expected data layers.
    """
    # Check if key dataframes are loaded
    assert hasattr(oracle, 'perf_2026')
    assert hasattr(oracle, 'fifa23')
    assert hasattr(oracle, 'pol_econ')
    assert not oracle.perf_2026.empty

def test_get_team_features_11_layers(oracle):
    """
    Test that get_team_features returns the target 11-layer schema.
    """
    features = oracle.get_team_features("Argentina", 2026)
    
    # Core performance & technical
    assert "elo" in features
    assert "technical_floor" in features
    assert "clutch_score_mean" in features
    assert "pressure_resistance_mean" in features
    
    # Psychopolitical
    assert "ppi" in features
    assert "uai" in features
    assert "political_stability" in features
    assert "gdp_per_capita" in features
    
    # Historical
    assert "wc_pedigree" in features

def test_build_2026_matrix_completeness(oracle):
    """
    Test that the 2026 matrix has no critical missing layers for the final model.
    """
    matrix = oracle.build_2026_matrix()
    expected_cols = ["team", "elo", "technical_floor", "clutch_score_mean", "ppi"]
    for col in expected_cols:
        assert col in matrix.columns
        assert not matrix[col].isnull().all()
