import pandas as pd
perf = pd.read_csv("data/processed/fifa_world_cup_2026_player_performance.csv")
print(perf['team'].unique())
