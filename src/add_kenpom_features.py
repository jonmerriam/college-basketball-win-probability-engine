from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

games = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
teams = pd.read_csv(DATA / "MTeams.csv")
mapping = pd.read_csv(DATA / "team_name_mapping.csv")
kenpom = pd.read_csv(DATA / "kenpom_2025.csv")

games = games[games["Season"] == 2025].copy()

# Add winner/loser names
games = games.merge(teams[["TeamID", "TeamName"]], left_on="WTeamID", right_on="TeamID", how="left")
games = games.rename(columns={"TeamName": "Winner"}).drop(columns=["TeamID"])

games = games.merge(teams[["TeamID", "TeamName"]], left_on="LTeamID", right_on="TeamID", how="left")
games = games.rename(columns={"TeamName": "Loser"}).drop(columns=["TeamID"])

translator = dict(zip(mapping["kaggle_name"], mapping["kenpom_name"]))

games["WinnerKP"] = games["Winner"].map(translator)
games["LoserKP"] = games["Loser"].map(translator)

features = [
    "Team",
    "Tempo-Adj",
    "Off. Efficiency-Adj",
    "Def. Efficiency-Adj",
]

kenpom_small = kenpom[features].copy()

winner_stats = kenpom_small.add_prefix("W_")
loser_stats = kenpom_small.add_prefix("L_")

dataset = games.merge(
    winner_stats,
    left_on="WinnerKP",
    right_on="W_Team",
    how="left"
)

dataset = dataset.merge(
    loser_stats,
    left_on="LoserKP",
    right_on="L_Team",
    how="left"
)

print(dataset.head())
print("\nRows:", len(dataset))
print("\nMissing values:")
print(dataset[[
    "W_Tempo-Adj",
    "W_Off. Efficiency-Adj",
    "W_Def. Efficiency-Adj",
    "L_Tempo-Adj",
    "L_Off. Efficiency-Adj",
    "L_Def. Efficiency-Adj",
]].isna().sum())

dataset.to_csv(DATA / "training_data_2025.csv", index=False)
print("\nSaved data/training_data_2025.csv")