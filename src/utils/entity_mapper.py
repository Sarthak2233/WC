import logging
from src.data.entity_resolver import resolve_country_name, get_iso3_code
import pycountry

logger = logging.getLogger(__name__)

# Dictionary to map various name formats to a canonical country name
# This is now a fallback and extension for entity_resolver
COUNTRY_NAME_MAP = {
    "USA": "United States",
    "United States": "United States",
    "United States of America": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "North Korea": "North Korea",
    "Korea DPR": "North Korea",
    "Czechia": "Czechia",
    "Czech Republic": "Czech Rep",
    "South Africa": "South Africa",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Turkiye": "Türkiye",
    "Turkey": "Türkiye",
    "Türkiye": "Türkiye",
    "DR Congo": "Congo DR",
    "Democratic Republic of the Congo": "Congo DR",
    "Congo, The Democratic Republic of the": "Congo DR",
    "Congo": "Congo DR",
    "Congo DR": "Congo DR",
    "Cabo Verde": "Cabo Verde",
    "Cape Verde": "Cabo Verde",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosina and Herzegovina": "Bosnia and Herzegovina",
    "Saudi Arabia": "Saudi Arabia",
    "Jordan": "Jordan",
    "Uzbekistan": "Uzbekistan",
}

def standardize_country_name(name: str) -> str:
    """Returns the canonical version of a country name using pycountry and custom maps."""
    if not name or not isinstance(name, str) or name.lower() == "nan" or name == "Unknown":
        return "Unknown"
    
    name = name.strip()
    
    # 1. Check if it's an ISO-3 alpha-3 code
    if len(name) == 3 and name.isupper():
        country = pycountry.countries.get(alpha_3=name)
        if country:
            return resolve_country_name(country.name)
            
    # 2. Use the robust resolver
    resolved = resolve_country_name(name)
    if resolved and resolved != name:
        # Map resolver result through manual overrides to ensure a single canonical form (e.g., 'Democratic Republic of the Congo' -> 'Congo DR')
        return COUNTRY_NAME_MAP.get(resolved, resolved)
        
    # 3. Fallback to manual map
    if name in COUNTRY_NAME_MAP:
        return COUNTRY_NAME_MAP[name]
    
    # 4. Normalized check
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
