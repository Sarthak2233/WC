import pycountry

# A hash map for O(1) resolution of common alternative and historical names.
# This ensures we adhere to the DSA_POLICY.md for efficient lookups.
CUSTOM_MAPPING = {
    # Alternative names
    "USA": "United States",
    "United States of America": "United States",
    "U.S.A.": "United States",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea", # Standardized internal representation
    "Korea DPR": "North Korea",
    "North Korea": "North Korea",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "IR Iran": "Iran",
    "Iran, Islamic Rep.": "Iran",
    "DR Congo": "Democratic Republic of the Congo",
    
    # Historical names (mapping to modern successor for continuity in ML models,
    # or kept separate if the user strictly prefers historical accuracy. 
    # For World Cup prediction, mapping to modern successor is usually preferred
    # to maintain continuous time-series data for the nation's footballing body).
    "West Germany": "Germany",
    "Zaire": "Democratic Republic of the Congo",
    "Yugoslavia": "Serbia", 
    "Serbia and Montenegro": "Serbia",
    "Soviet Union": "Russia",
    "Czechoslovakia": "Czechia",
    "Czech Republic": "Czechia",
    "Dutch East Indies": "Indonesia",
}

def resolve_country_name(name: str) -> str:
    """
    Resolves a country name to a standardized modern name.
    Time Complexity: O(1) hash map lookup.
    """
    if not name:
        return ""
        
    name_clean = name.strip()
    
    # Fast path: check custom mappings
    if name_clean in CUSTOM_MAPPING:
        return CUSTOM_MAPPING[name_clean]
        
    # Standard pycountry lookup
    try:
        # First try exact match
        country = pycountry.countries.get(name=name_clean)
        if country:
            return country.name
            
        # Try common name
        country = pycountry.countries.get(common_name=name_clean)
        if country:
            return getattr(country, "common_name", country.name)
            
        # Try official name
        country = pycountry.countries.get(official_name=name_clean)
        if country:
            return getattr(country, "official_name", country.name)
            
    except Exception:
        pass
        
    # If all fails, return the cleaned input to be logged or handled upstream
    return name_clean


# FIPS 10-4 to ISO3 mapping for GDELT
FIPS_TO_ISO3 = {
    "AF": "AFG", "AL": "ALB", "AG": "DZA", "AN": "AND", "AO": "AGO", "AR": "ARG", "AM": "ARM", "AS": "AUS",
    "AU": "AUT", "AJ": "AZE", "BA": "BHR", "BG": "BGD", "BE": "BEL", "BH": "BLZ", "BN": "BEN", "BT": "BTN",
    "BL": "BOL", "BK": "BIH", "BC": "BWA", "BR": "BRA", "BX": "BRN", "BU": "BGR", "UV": "BFA", "BY": "BDI",
    "CB": "KHM", "CM": "CMR", "CA": "CAN", "CT": "CAF", "CD": "TCD", "CI": "CHL", "CH": "CHN", "CO": "COL",
    "CN": "COM", "CG": "COD", "CF": "COG", "CS": "CRI", "IV": "CIV", "HR": "HRV", "CU": "CUB", "CY": "CYP",
    "EZ": "CZE", "DA": "DNK", "DJ": "DJI", "DO": "DMA", "DR": "DOM", "EC": "ECHO", "EG": "EGY", "ES": "SLV",
    "EK": "GNQ", "ER": "ERI", "EN": "EST", "ET": "ETH", "FJ": "FJI", "FI": "FIN", "FR": "FRA", "GB": "GAB",
    "GA": "GMB", "GG": "GEO", "GM": "DEU", "GH": "GHA", "GR": "GRC", "GJ": "GRD", "GT": "GTM", "GV": "GIN",
    "PU": "GNB", "GY": "GUY", "HA": "HTI", "HO": "HND", "HU": "HUN", "IC": "ISL", "IN": "IND", "ID": "IDN",
    "IR": "IRN", "IZ": "IRQ", "EI": "IRL", "IS": "ISR", "IT": "ITA", "JM": "JAM", "JA": "JPN", "JO": "JOR",
    "KZ": "KAZ", "KE": "KEN", "KR": "KIR", "KN": "PRK", "KS": "KOR", "KU": "KWT", "KG": "KGZ", "LA": "LAO",
    "LG": "LVA", "LE": "LBN", "LT": "LSO", "LI": "LBR", "LY": "LBY", "LS": "LIE", "LH": "LTU", "LU": "LUX",
    "MK": "MKD", "MA": "MDG", "MI": "MWI", "MY": "MYS", "MV": "MDV", "ML": "MLI", "MT": "MLT", "RM": "MHL",
    "MR": "MRT", "MP": "MUS", "MX": "MEX", "FM": "FSM", "MD": "MDA", "MN": "MCO", "MN": "MNG", "MJ": "MNE",
    "MO": "MAR", "MZ": "MOZ", "BM": "MMR", "WA": "NAM", "NR": "NRU", "NP": "NPL", "NL": "NLD", "NZ": "NZL",
    "NU": "NIC", "NG": "NER", "NI": "NGA", "NO": "NOR", "MU": "OMN", "PK": "PAK", "PS": "PLW", "PM": "PAN",
    "PP": "PNG", "PA": "PRY", "PE": "PER", "RP": "PHL", "PL": "POL", "PO": "PRT", "QA": "QAT", "RO": "ROU",
    "RS": "RUS", "RW": "RWA", "SC": "KNA", "ST": "LCA", "VC": "VCT", "WS": "WSM", "SM": "SMR", "TP": "STP",
    "SA": "SAU", "SG": "SEN", "RI": "SRB", "SE": "SYC", "SL": "SLE", "SN": "SGP", "LO": "SVK", "SI": "SVN",
    "BP": "SLB", "SO": "SOM", "SF": "ZAF", "OD": "SSD", "SP": "ESP", "CE": "LKA", "SU": "SDN", "NS": "SUR",
    "WZ": "SWZ", "SW": "SWE", "SZ": "CHE", "SY": "SYR", "TW": "TWN", "TI": "TJK", "TZ": "TZA", "TH": "THA",
    "TO": "TGO", "TN": "TUN", "TU": "TUR", "TX": "TKM", "TV": "TUV", "UG": "UGA", "UP": "UKR", "AE": "ARE",
    "UK": "GBR", "US": "USA", "UY": "URY", "UZ": "UZB", "NH": "VUT", "VT": "VAT", "VE": "VEN", "VM": "VNM",
    "YM": "YEM", "ZA": "ZMB", "ZI": "ZWE"
}

def resolve_fips_to_iso3(fips: str) -> str | None:
    return FIPS_TO_ISO3.get(fips.upper()) if fips else None


def get_iso3_code(name: str) -> str | None:
    """
    Gets the ISO-3166-1 alpha-3 code for a country name or alternative name.
    Returns None if not found.
    Time Complexity: O(1) or O(K) where K is number of countries depending on pycountry.
    """
    if not name:
        return None
        
    standard_name = resolve_country_name(name)
    
    try:
        # Check if name is already an ISO3 code
        country = pycountry.countries.get(alpha_3=standard_name.upper())
        if country:
            return country.alpha_3
            
        country = pycountry.countries.get(name=standard_name)
        if country:
            return country.alpha_3
            
        # Some countries in pycountry use 'common_name'
        # Let's search
        for c in pycountry.countries:
            if getattr(c, "common_name", "") == standard_name or c.name == standard_name:
                return c.alpha_3
                
    except Exception:
        pass
        
    return None
