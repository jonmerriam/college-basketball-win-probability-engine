import os
from io import StringIO
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from kenpompy.utils import login


load_dotenv()

EMAIL = os.getenv("KENPOM_EMAIL")
PASSWORD = os.getenv("KENPOM_PASSWORD")

if EMAIL is None or PASSWORD is None:
    raise Exception("Missing KenPom credentials in .env")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

browser = login(EMAIL, PASSWORD)


CLEAN_COLUMNS = [
    "Rank",
    "Team",
    "Conference",
    "Record",
    "AdjEM",
    "AdjOE",
    "AdjOE_Rank",
    "AdjDE",
    "AdjDE_Rank",
    "AdjTempo",
    "AdjTempo_Rank",
    "Luck",
    "Luck_Rank",
    "SOS_AdjEM",
    "SOS_AdjEM_Rank",
    "SOS_OppOE",
    "SOS_OppOE_Rank",
    "SOS_OppDE",
    "SOS_OppDE_Rank",
    "NCSOS_AdjEM",
    "NCSOS_AdjEM_Rank",
]


for year in range(2021, 2026):
    print(f"\nPulling ratings table for {year}...")

    url = f"https://kenpom.com/index.php?y={year}"
    html = browser.get(url).text

    tables = pd.read_html(StringIO(html))

    print(f"Found {len(tables)} tables")

    df = max(tables, key=lambda t: t.shape[0] * t.shape[1]).copy()

    if df.shape[1] != len(CLEAN_COLUMNS):
        raise ValueError(
            f"Expected {len(CLEAN_COLUMNS)} columns, got {df.shape[1]} columns for {year}"
        )

    df.columns = CLEAN_COLUMNS

    # Remove repeated header rows or weird non-team rows
    df = df[df["Rank"].astype(str).str.isnumeric()].copy()

    # Convert numeric columns
    numeric_cols = [
        "Rank",
        "AdjEM",
        "AdjOE",
        "AdjOE_Rank",
        "AdjDE",
        "AdjDE_Rank",
        "AdjTempo",
        "AdjTempo_Rank",
        "Luck",
        "Luck_Rank",
        "SOS_AdjEM",
        "SOS_AdjEM_Rank",
        "SOS_OppOE",
        "SOS_OppOE_Rank",
        "SOS_OppDE",
        "SOS_OppDE_Rank",
        "NCSOS_AdjEM",
        "NCSOS_AdjEM_Rank",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    filename = DATA / f"kenpom_ratings_{year}.csv"
    df.to_csv(filename, index=False)

    print(f"Saved {filename}")
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())
    print(df.head())