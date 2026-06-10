import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SquadScraper:
    """
    Scrapes World Cup squad data from Wikipedia.
    """
    
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # If the 2026 squads page doesn't exist yet, we will fallback to a generic URL
        # or handle 404s gracefully.
        
    def fetch_page(self, url: str) -> str:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [403, 429]:
                logger.warning(f"Received {e.response.status_code} for {url}. Skipping.")
                return ""
            logger.error(f"Failed to fetch {url}: {e}")
            return ""
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return ""
            
    def parse_squads(self, html: str) -> pd.DataFrame:
        """
        Parses the standard Wikipedia FIFA World Cup squads table format.
        """
        if not html:
            return pd.DataFrame()
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # In a real Wikipedia World Cup squads page, each team usually has a heading (<h3> or <h2>)
        # followed by a table with class="sortable"
        
        squads = []
        
        # This is a very simplified parsing logic since the 2026 page format isn't final
        for header in soup.find_all(['h2', 'h3']):
            headline = header.find(class_='mw-headline')
            if not headline:
                continue
                
            country = headline.text.strip()
            
            # Find the next table
            table = header.find_next_sibling('table', class_='sortable')
            if not table:
                continue
                
            # Parse rows
            for row in table.find_all('tr')[1:]: # Skip header
                cols = row.find_all(['th', 'td'])
                if len(cols) >= 6:
                    # typical columns: No., Pos., Player, Date of birth (age), Caps, Goals, Club
                    try:
                        pos = cols[1].text.strip()
                        name = cols[2].text.strip()
                        # Clean up references like [1]
                        name = name.split('[')[0].strip()
                        
                        caps = cols[4].text.strip()
                        caps = int(caps) if caps.isdigit() else 0
                        
                        goals = cols[5].text.strip()
                        goals = int(goals) if goals.isdigit() else 0
                        
                        squads.append({
                            "country": country,
                            "position": pos,
                            "name": name,
                            "caps": caps,
                            "goals": goals
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse row for {country}: {e}")
                        continue
                        
        return pd.DataFrame(squads)
        
    def scrape_2026_squads(self) -> pd.DataFrame:
        """
        Main entry point for scraping 2026 squads.
        """
        html = self.fetch_page(self.base_url)
        return self.parse_squads(html)
