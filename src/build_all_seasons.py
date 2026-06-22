from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

START_SEASON = 2021
END_SEASON = 2025

FEATURES = [
    "AdjTempo",
    "AdjOE",
    "Off-eFG%",
    "Off-TO%",
    "Off-OR%",
    "Off-FTRate",
    "AdjDE",
    "Def-eFG%",
    "Def-TO%",
    "Def-OR%",
    "Def-FTRate",
]

games = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
teams = pd.read_csv(DATA / "MTeams.csv")
mapping = pd.read_csv(DATA / "team_name_mapping.csv")

translator = dict(zip(mapping["kaggle_name"], mapping["kenpom_name"]))

all_rows = []


def loser_location(wloc):
    if wloc == "H":
        return "A"
    if wloc == "A":
        return "H"
    return "N"


def clean_feature_name(feature):
    return (
        feature
        .replace("%", "Pct")
        .replace("-", "_")
    )


for season in range(START_SEASON, END_SEASON + 1):
    print(f"Building season {season}...")

    season_games = games[games["Season"] == season].copy()
    kenpom = pd.read_csv(DATA / f"kenpom_fourfactors_{season}.csv")

    season_games = season_games.merge(
        teams[["TeamID", "TeamName"]],
        left_on="WTeamID",
        right_on="TeamID",
        how="left",
    ).rename(columns={"TeamName": "Winner"}).drop(columns=["TeamID"])

    season_games = season_games.merge(
        teams[["TeamID", "TeamName"]],
        left_on="LTeamID",
        right_on="TeamID",
        how="left",
    ).rename(columns={"TeamName": "Loser"}).drop(columns=["TeamID"])

    season_games["WinnerKP"] = season_games["Winner"].map(translator)
    season_games["LoserKP"] = season_games["Loser"].map(translator)

    kp = kenpom[["Team", "Conference"] + FEATURES].copy()

    winner_stats = kp.add_prefix("W_")
    loser_stats = kp.add_prefix("L_")

    df = season_games.merge(
        winner_stats,
        left_on="WinnerKP",
        right_on="W_Team",
        how="left",
    )

    df = df.merge(
        loser_stats,
        left_on="LoserKP",
        right_on="L_Team",
        how="left",
    )

    wins = pd.DataFrame({
        "Season": df["Season"],
        "TeamA": df["WinnerKP"],
        "TeamB": df["LoserKP"],
        "TeamA_Won": 1,
        "ScoreDiff": df["WScore"] - df["LScore"],

        "A_Loc": df["WLoc"],
        "A_Conference": df["W_Conference"],
        "B_Conference": df["L_Conference"],
    })

    losses = pd.DataFrame({
        "Season": df["Season"],
        "TeamA": df["LoserKP"],
        "TeamB": df["WinnerKP"],
        "TeamA_Won": 0,
        "ScoreDiff": df["LScore"] - df["WScore"],

        "A_Loc": df["WLoc"].apply(loser_location),
        "A_Conference": df["L_Conference"],
        "B_Conference": df["W_Conference"],
    })

    for feature in FEATURES:
        clean_name = clean_feature_name(feature)

        wins[f"A_{clean_name}"] = df[f"W_{feature}"]
        wins[f"B_{clean_name}"] = df[f"L_{feature}"]

        losses[f"A_{clean_name}"] = df[f"L_{feature}"]
        losses[f"B_{clean_name}"] = df[f"W_{feature}"]

    all_rows.append(wins)
    all_rows.append(losses)


training = pd.concat(all_rows, ignore_index=True)

print("\nMissing values before drop:")
print(training.isna().sum())

training = training.dropna().copy()

print("\nMissing values after drop:")
print(training.isna().sum())

print("\nClass balance:")
print(training["TeamA_Won"].value_counts())

training.to_csv(DATA / "modeling_dataset.csv", index=False)

print("\nSaved data/modeling_dataset.csv")
print("Rows:", len(training))
print(training.head())
print(training.columns.tolist())