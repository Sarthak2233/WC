import pytest
from src.data.entity_resolver import resolve_country_name, get_iso3_code

def test_resolve_standard_names():
    """Test resolving standard country names."""
    assert resolve_country_name("United States") == "United States"
    assert resolve_country_name("Brazil") == "Brazil"

def test_resolve_fuzzy_names():
    """Test resolving common variations and alternative names."""
    assert resolve_country_name("USA") == "United States"
    assert resolve_country_name("Korea Republic") == "South Korea"
    assert resolve_country_name("Korea DPR") == "North Korea"
    assert resolve_country_name("Ivory Coast") == "Côte d'Ivoire"

def test_resolve_historical_names():
    """Test resolving historical names if mapped to modern equivalents."""
    assert resolve_country_name("West Germany") == "Germany"
    assert resolve_country_name("Zaire") == "Democratic Republic of the Congo"

def test_get_iso3_code():
    """Test getting ISO3 codes for standardized names."""
    assert get_iso3_code("United States") == "USA"
    assert get_iso3_code("Brazil") == "BRA"
    assert get_iso3_code("Germany") == "DEU"
    assert get_iso3_code("Unknown") is None
