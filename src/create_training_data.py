from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# --------------------
# Load files
# --------------------

games = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
teams = pd.read_csv(DATA / "MTeams.csv")
mapping = pd.read_csv(DATA / "team_name_mapping.csv")
kenpom = pd.read_csv(DATA / "kenpom_2025.csv")

# --------------------
# Only use 2025 games
# --------------------

games = games[games["Season"] == 2025].copy()

# --------------------
# Add team names
# --------------------

games = games.merge(
    teams[["TeamID", "TeamName"]],
    left_on="WTeamID",
    right_on="TeamID",
    how="left"
)

games = games.rename(
    columns={"TeamName": "Winner"}
)

games = games.drop(columns=["TeamID"])

games = games.merge(
    teams[["TeamID", "TeamName"]],
    left_on="LTeamID",
    right_on="TeamID",
    how="left"
)

games = games.rename(
    columns={"TeamName": "Loser"}
)

games = games.drop(columns=["TeamID"])

# --------------------
# Translate names
# --------------------

translator = dict(
    zip(
        mapping["kaggle_name"],
        mapping["kenpom_name"]
    )
)

games["WinnerKP"] = games["Winner"].map(translator)

games["LoserKP"] = games["Loser"].map(translator)

# --------------------
# Keep useful columns
# --------------------

dataset = games[
    [
        "Season",
        "WinnerKP",
        "LoserKP",
        "WScore",
        "LScore",
        "WLoc"
    ]
]

print(dataset.head())
print()
print("Rows:", len(dataset))
print("Missing WinnerKP:", dataset["WinnerKP"].isna().sum())
print("Missing LoserKP:", dataset["LoserKP"].isna().sum())