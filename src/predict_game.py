from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

MODEL_PATH = MODELS / "win_probability_model.pkl"
KENPOM_PATH = DATA / "kenpom_fourfactors_2025.csv"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]

kenpom = pd.read_csv(KENPOM_PATH)


def get_team_stats(team_name):
    exact = kenpom[kenpom["Team"].str.lower() == team_name.lower()]

    if not exact.empty:
        return exact.iloc[0]

    matches = kenpom[
        kenpom["Team"].str.lower().str.contains(team_name.lower(), na=False)
    ]

    if not matches.empty:
        print("\nTeam not found exactly. Did you mean one of these?")
        for name in matches["Team"].tolist():
            print(f"- {name}")

    raise ValueError(f"Team not found: {team_name}")


def build_matchup_row(team_a, team_b, location):
    a = get_team_stats(team_a)
    b = get_team_stats(team_b)

    row = {
        "A_Loc": location,
        "A_Conference": a["Conference"],
        "B_Conference": b["Conference"],

        "A_AdjTempo": a["AdjTempo"],
        "A_AdjOE": a["AdjOE"],
        "A_Off_eFGPct": a["Off-eFG%"],
        "A_Off_TOPct": a["Off-TO%"],
        "A_Off_ORPct": a["Off-OR%"],
        "A_Off_FTRate": a["Off-FTRate"],
        "A_AdjDE": a["AdjDE"],
        "A_Def_eFGPct": a["Def-eFG%"],
        "A_Def_TOPct": a["Def-TO%"],
        "A_Def_ORPct": a["Def-OR%"],
        "A_Def_FTRate": a["Def-FTRate"],

        "B_AdjTempo": b["AdjTempo"],
        "B_AdjOE": b["AdjOE"],
        "B_Off_eFGPct": b["Off-eFG%"],
        "B_Off_TOPct": b["Off-TO%"],
        "B_Off_ORPct": b["Off-OR%"],
        "B_Off_FTRate": b["Off-FTRate"],
        "B_AdjDE": b["AdjDE"],
        "B_Def_eFGPct": b["Def-eFG%"],
        "B_Def_TOPct": b["Def-TO%"],
        "B_Def_ORPct": b["Def-OR%"],
        "B_Def_FTRate": b["Def-FTRate"],
    }

    matchup = pd.DataFrame([row])

    matchup = pd.get_dummies(
        matchup,
        columns=categorical_features,
        drop_first=False,
    )

    matchup = matchup.reindex(columns=feature_columns, fill_value=0)

    return matchup


def predict_game(team_a, team_b, location):
    matchup = build_matchup_row(team_a, team_b, location)

    prob_a = model.predict_proba(matchup)[0][1]
    prob_b = 1 - prob_a

    print("\n--- Win Probability Simulation ---")
    print(f"{team_a}: {prob_a:.1%}")
    print(f"{team_b}: {prob_b:.1%}")

    if prob_a > prob_b:
        print(f"Predicted winner: {team_a}")
    else:
        print(f"Predicted winner: {team_b}")


if __name__ == "__main__":
    print("\nLocation options:")
    print("H = Team A is home")
    print("A = Team A is away")
    print("N = Neutral site")

    team_a = input("\nEnter Team A: ")
    team_b = input("Enter Team B: ")
    location = input("Location for Team A (H/A/N): ").upper().strip()

    if location not in ["H", "A", "N"]:
        raise ValueError("Location must be H, A, or N")

    predict_game(team_a, team_b, location)
    