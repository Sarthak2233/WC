import logging

logger = logging.getLogger(__name__)

# Dictionary to map various name formats to a canonical country name
COUNTRY_NAME_MAP = {
    "USA": "United States",
    "United States of America": "United States",
    "South Korea": "Korea Republic",
    "Republic of Korea": "Korea Republic",
    "Czechia": "Czech Republic",
    "South Africa": "South Africa",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Turkiye": "Türkiye",
    "Turkey": "Türkiye",
    "DR Congo": "Congo DR",
    "Democratic Republic of the Congo": "Congo DR",
    "Cabo Verde": "Cabo Verde",
    "Cape Verde": "Cabo Verde",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosina and Herzegovina": "Bosnia and Herzegovina",
    "Netherlands": "Netherlands",
    "Holland": "Netherlands",
    "England": "England",
    "Scotland": "Scotland",
    "Wales": "Wales",
    "Northern Ireland": "Northern Ireland",
    "Spain": "Spain",
    "Argentina": "Argentina",
    "Brazil": "Brazil",
    "Germany": "Germany",
    "France": "France",
    "Italy": "Italy",
    "Portugal": "Portugal",
    "Mexico": "Mexico",
    "Morocco": "Morocco",
    "Senegal": "Senegal",
    "Switzerland": "Switzerland",
    "Ecuador": "Ecuador",
    "Canada": "Canada",
    "Australia": "Australia",
    "Japan": "Japan",
    "Qatar": "Qatar",
    "Saudi Arabia": "Saudi Arabia",
    "Iran": "Iran",
    "Egypt": "Egypt",
    "Ghana": "Ghana",
    "Uruguay": "Uruguay",
    "Colombia": "Colombia",
    "Croatia": "Croatia",
    "Belgium": "Belgium",
    "Denmark": "Denmark",
    "Norway": "Norway",
    "Sweden": "Sweden",
    "Austria": "Austria",
    "Poland": "Poland",
    "Ukraine": "Ukraine",
    "Russia": "Russia",
    "Serbia": "Serbia",
    "Algeria": "Algeria",
    "Tunisia": "Tunisia",
    "Panama": "Panama",
    "Iraq": "Iraq",
    "Jordan": "Jordan",
    "Uzbekistan": "Uzbekistan",
}

def standardize_country_name(name: str) -> str:
    """Returns the canonical version of a country name."""
    if not name or not isinstance(name, str):
        return "Unknown"
    
    name = name.strip()
    # Check map first
    if name in COUNTRY_NAME_MAP:
        return COUNTRY_NAME_MAP[name]
    
    # Try normalized check (lowercase/no accents - simplified here)
    normalized_name = name.encode('ascii', 'ignore').decode('ascii').lower()
    
    for key, val in COUNTRY_NAME_MAP.items():
        if key.lower() == normalized_name or val.lower() == normalized_name:
            return val
            
    return name

def standardize_player_name(name: str) -> str:
    """Strips and normalizes player names."""
    if not name or not isinstance(name, str):
        return "Unknown"
    return name.strip()
