from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

games = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
teams = pd.read_csv(DATA / "MTeams.csv")
kenpom = pd.read_csv(DATA / "kenpom_2025.csv")

# only 2025 for now
games = games[games["Season"] == 2025].copy()

# add winning team name
games = games.merge(
    teams[["TeamID", "TeamName"]],
    left_on="WTeamID",
    right_on="TeamID",
    how="left"
).rename(columns={"TeamName": "WTeamName"}).drop(columns=["TeamID"])

# add losing team name
games = games.merge(
    teams[["TeamID", "TeamName"]],
    left_on="LTeamID",
    right_on="TeamID",
    how="left"
).rename(columns={"TeamName": "LTeamName"}).drop(columns=["TeamID"])

print(games[["Season", "WTeamName", "WScore", "LTeamName", "LScore", "WLoc"]].head(20))

print("\nKaggle teams example:")
print(sorted(games["WTeamName"].dropna().unique())[:30])

print("\nKenPom teams example:")
print(sorted(kenpom["Team"].dropna().unique())[:30])