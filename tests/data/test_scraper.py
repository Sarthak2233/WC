import pytest
import pandas as pd
import requests
from unittest.mock import MagicMock
from src.data.scraper import SquadScraper

def test_parse_squads():
    scraper = SquadScraper()
    
    # Mock HTML simulating Wikipedia format
    html = """
    <h3><span class="mw-headline">Brazil</span></h3>
    <table class="sortable">
        <tr><th>No.</th><th>Pos.</th><th>Player</th><th>DOB</th><th>Caps</th><th>Goals</th><th>Club</th></tr>
        <tr><td>1</td><td>GK</td><td>Alisson</td><td>1992</td><td>61</td><td>0</td><td>Liverpool</td></tr>
        <tr><td>10</td><td>FW</td><td>Neymar [1]</td><td>1992</td><td>124</td><td>77</td><td>Al Hilal</td></tr>
    </table>
    """
    
    df = scraper.parse_squads(html)
    assert not df.empty
    assert len(df) == 2
    assert df.iloc[0]["name"] == "Alisson"
    assert df.iloc[0]["country"] == "Brazil"
    assert df.iloc[1]["name"] == "Neymar"
    assert df.iloc[1]["caps"] == 124
    assert df.iloc[1]["goals"] == 77

def test_fetch_page_handles_403_429(monkeypatch):
    scraper = SquadScraper()
    
    # Mock requests.get to raise HTTPError with 403
    mock_get = MagicMock()
    mock_get.raise_for_status.side_effect = requests.exceptions.HTTPError(response=MagicMock(status_code=403))
    monkeypatch.setattr("requests.get", MagicMock(return_value=mock_get))
    
    result = scraper.fetch_page("http://test.com")
    assert result == ""
    
    # Mock requests.get to raise HTTPError with 429
    mock_get.raise_for_status.side_effect = requests.exceptions.HTTPError(response=MagicMock(status_code=429))
    monkeypatch.setattr("requests.get", MagicMock(return_value=mock_get))
    
    result = scraper.fetch_page("http://test.com")
    assert result == ""
