# Data Lineage: World Cup 2026 Oracle

| Layer | Dataset | Raw File Source | Loader/Scraper | Processed Output |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Tournaments | GitHub (Fjelstul) | FootballLoader | world_cups.csv |
| 1 | Matches | GitHub (Fjelstul) | FootballLoader | matches.csv |
| 1 | Squads (Hist) | GitHub (Fjelstul) | FootballLoader | squads.csv |
| 1 | Squads (2026) | Wikipedia | SquadScraper | master_squads_2026.csv |
| 2 | Performance | Performance Metrics | N/A (Static) | fifa_world_cup_2026_player_performance.csv |
| 5/6| Pol/Econ | World Bank API | PoliticalLoader | political_economic.csv |
| 7 | Conflict | UCDP (Mirror) | ConflictLoader | conflict_data.csv |
| 8 | Hofstede | GitHub (Plotly) | CultureLoader | culture_happiness.csv |
| 9 | Happiness | GitHub (pplonski) | CultureLoader | culture_happiness.csv |
| 10 | Psyche | Perfor. Metrics | PsycheLoader | player_psychology.csv |
| 11 | Narratives | N/A | N/A | N/A |
