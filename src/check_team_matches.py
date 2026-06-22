from pathlib import Path
import pandas as pd
from difflib import get_close_matches

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

teams = pd.read_csv(DATA / "MTeams.csv")
# Keep only teams active through 2025
teams = teams[teams["LastD1Season"] == 2025].copy()

kenpom = pd.read_csv(DATA / "kenpom_2025.csv")

kaggle_names = sorted(teams["TeamName"].dropna().unique())
kenpom_names = sorted(kenpom["Team"].dropna().unique())

rows = []

for name in kaggle_names:
    if name in kenpom_names:
        match = name
        status = "exact"
    else:
        close = get_close_matches(name, kenpom_names, n=1, cutoff=0.65)
        match = close[0] if close else ""
        status = "guess" if close else "missing"

    rows.append({
        "kaggle_name": name,
        "kenpom_name": match,
        "status": status
    })

mapping = pd.DataFrame(rows)

mapping.to_csv(DATA / "team_name_mapping.csv", index=False)

print(mapping["status"].value_counts())
print("\nReview these guesses/missing values:")
print(mapping[mapping["status"] != "exact"].head(80))
print("\nSaved data/team_name_mapping.csv")