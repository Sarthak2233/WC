import pandas as pd
perf = pd.read_csv("data/processed/2026_only/fifa_world_cup_2026_player_performance.csv")
print(perf['team'].unique())
