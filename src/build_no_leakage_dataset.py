from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FEATURES = [
    "Tempo-Adj",
    "Off. Efficiency-Adj",
    "Def. Efficiency-Adj",
]

# Predict each season using the PRIOR season's KenPom ratings
START_SEASON = 2022
END_SEASON = 2025

games = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
teams = pd.read_csv(DATA / "MTeams.csv")
mapping = pd.read_csv(DATA / "team_name_mapping.csv")

translator = dict(zip(mapping["kaggle_name"], mapping["kenpom_name"]))

all_rows = []

for season in range(START_SEASON, END_SEASON + 1):
    print(f"Building season {season} using KenPom {season - 1}...")

    season_games = games[games["Season"] == season].copy()
    kenpom = pd.read_csv(DATA / f"kenpom_{season - 1}.csv")

    season_games = season_games.merge(
        teams[["TeamID", "TeamName"]],
        left_on="WTeamID",
        right_on="TeamID",
        how="left"
    ).rename(columns={"TeamName": "Winner"}).drop(columns=["TeamID"])

    season_games = season_games.merge(
        teams[["TeamID", "TeamName"]],
        left_on="LTeamID",
        right_on="TeamID",
        how="left"
    ).rename(columns={"TeamName": "Loser"}).drop(columns=["TeamID"])

    season_games["WinnerKP"] = season_games["Winner"].map(translator)
    season_games["LoserKP"] = season_games["Loser"].map(translator)

    kp = kenpom[["Team"] + FEATURES].copy()

    winner_stats = kp.add_prefix("W_")
    loser_stats = kp.add_prefix("L_")

    df = season_games.merge(
        winner_stats,
        left_on="WinnerKP",
        right_on="W_Team",
        how="left"
    )

    df = df.merge(
        loser_stats,
        left_on="LoserKP",
        right_on="L_Team",
        how="left"
    )

    wins = pd.DataFrame({
        "Season": df["Season"],
        "TeamA": df["WinnerKP"],
        "TeamB": df["LoserKP"],
        "TeamA_Won": 1,
        "ScoreDiff": df["WScore"] - df["LScore"],
        "Loc": df["WLoc"],

        "A_Tempo": df["W_Tempo-Adj"],
        "A_OffEff": df["W_Off. Efficiency-Adj"],
        "A_DefEff": df["W_Def. Efficiency-Adj"],

        "B_Tempo": df["L_Tempo-Adj"],
        "B_OffEff": df["L_Off. Efficiency-Adj"],
        "B_DefEff": df["L_Def. Efficiency-Adj"],
    })

    losses = pd.DataFrame({
        "Season": df["Season"],
        "TeamA": df["LoserKP"],
        "TeamB": df["WinnerKP"],
        "TeamA_Won": 0,
        "ScoreDiff": df["LScore"] - df["WScore"],
        "Loc": df["WLoc"],

        "A_Tempo": df["L_Tempo-Adj"],
        "A_OffEff": df["L_Off. Efficiency-Adj"],
        "A_DefEff": df["L_Def. Efficiency-Adj"],

        "B_Tempo": df["W_Tempo-Adj"],
        "B_OffEff": df["W_Off. Efficiency-Adj"],
        "B_DefEff": df["W_Def. Efficiency-Adj"],
    })

    all_rows.append(wins)
    all_rows.append(losses)

dataset = pd.concat(all_rows, ignore_index=True)

print("\nBefore dropping missing:")
print(dataset.isna().sum())

dataset = dataset.dropna().copy()

print("\nAfter dropping missing:")
print(dataset.isna().sum())

print("\nClass balance:")
print(dataset["TeamA_Won"].value_counts())

dataset.to_csv(DATA / "modeling_dataset_no_leakage.csv", index=False)

print("\nSaved data/modeling_dataset_no_leakage.csv")
print("Rows:", len(dataset))
print(dataset.head())