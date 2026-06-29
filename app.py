from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

MODEL_PATH = MODELS / "win_probability_model.pkl"
KENPOM_PATH = DATA / "kenpom_fourfactors_2025.csv"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_columns = bundle["feature_columns"]
categorical_features = bundle["categorical_features"]

kenpom = pd.read_csv(KENPOM_PATH)

def clean_dataframe_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prevents Streamlit / PyArrow display errors caused by mixed object columns.
    """
    cleaned = df.copy()

    cleaned.columns = cleaned.columns.astype(str)

    for column in cleaned.columns:
        if cleaned[column].dtype == "object":
            cleaned[column] = cleaned[column].astype(str)

    return cleaned

def build_matchup_row(team_a, team_b, location):
    a = kenpom[kenpom["Team"] == team_a].iloc[0]
    b = kenpom[kenpom["Team"] == team_b].iloc[0]

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

    return matchup, a, b


def predict_game(team_a, team_b, location):
    matchup, a, b = build_matchup_row(team_a, team_b, location)

    prob_a = model.predict_proba(matchup)[0][1]
    prob_b = 1 - prob_a

    return prob_a, prob_b, a, b


st.set_page_config(
    page_title="College Basketball Win Probability Engine",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 College Basketball Win Probability Engine (2024-25)")

st.write(
    "Simulate a Division I college basketball matchup using KenPom four-factor "
    "metrics, conference data, and game location."
)

teams = sorted(kenpom["Team"].dropna().unique())

col1, col2, col3 = st.columns(3)

with col1:
    team_a = st.selectbox(
        "Team A",
        teams,
        index=teams.index("Duke") if "Duke" in teams else 0,
    )

with col2:
    team_b = st.selectbox(
        "Team B",
        teams,
        index=teams.index("North Carolina") if "North Carolina" in teams else 1,
    )

with col3:
    location_label = st.selectbox(
        "Location",
        [
            "Team A home",
            "Team A away",
            "Neutral site",
        ],
    )

location_map = {
    "Team A home": "H",
    "Team A away": "A",
    "Neutral site": "N",
}

location = location_map[location_label]

if team_a == team_b:
    st.warning("Choose two different teams.")
else:
    prob_a, prob_b, a, b = predict_game(team_a, team_b, location)

    st.subheader("Win Probability")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(team_a, f"{prob_a:.1%}")

    with col2:
        st.metric(team_b, f"{prob_b:.1%}")

    winner = team_a if prob_a > prob_b else team_b
    st.success(f"Predicted winner: {winner}")

    st.subheader("Matchup Comparison")

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Conference",
                "Adjusted Tempo",
                "Adjusted Offensive Efficiency",
                "Adjusted Defensive Efficiency",
                "Offensive eFG%",
                "Offensive TO%",
                "Offensive OR%",
                "Offensive FT Rate",
                "Defensive eFG%",
                "Defensive TO%",
                "Defensive OR%",
                "Defensive FT Rate",
            ],
            team_a: [
                a["Conference"],
                a["AdjTempo"],
                a["AdjOE"],
                a["AdjDE"],
                a["Off-eFG%"],
                a["Off-TO%"],
                a["Off-OR%"],
                a["Off-FTRate"],
                a["Def-eFG%"],
                a["Def-TO%"],
                a["Def-OR%"],
                a["Def-FTRate"],
            ],
            team_b: [
                b["Conference"],
                b["AdjTempo"],
                b["AdjOE"],
                b["AdjDE"],
                b["Off-eFG%"],
                b["Off-TO%"],
                b["Off-OR%"],
                b["Off-FTRate"],
                b["Def-eFG%"],
                b["Def-TO%"],
                b["Def-OR%"],
                b["Def-FTRate"],
            ],
        }
    )

    st.dataframe(
    clean_dataframe_for_streamlit(comparison),
    use_container_width=True
)

    st.caption(
        "Created by Jonathan Merriam. Model trained on 2021–2024 NCAA regular-season results and evaluated "
        "on 2025 games. Current simulator uses 2025 KenPom four-factor profiles."
    )