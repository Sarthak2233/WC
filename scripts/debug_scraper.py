import requests
from bs4 import BeautifulSoup
import urllib.parse

def debug_scraper():
    player_name = "Lionel Messi"
    base_url = "https://www.footballcritic.com"
    search_url = f"{base_url}/search?query={urllib.parse.quote(player_name)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Debug print the page structure to find the right selector
    print("Page Title:", soup.title.text if soup.title else "No title")
    # Print the first 500 characters of the page to get an idea of structure
    print("Page snippet:", response.text[:500])

if __name__ == "__main__":
    debug_scraper()
