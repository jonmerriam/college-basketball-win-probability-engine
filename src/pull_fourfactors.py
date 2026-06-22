import os
from pathlib import Path

from dotenv import load_dotenv
from kenpompy.utils import login
import kenpompy.summary as kp


load_dotenv()

EMAIL = os.getenv("KENPOM_EMAIL")
PASSWORD = os.getenv("KENPOM_PASSWORD")

if EMAIL is None or PASSWORD is None:
    raise Exception("Missing KenPom credentials in .env")


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

browser = login(EMAIL, PASSWORD)

for year in range(2021, 2026):
    print(f"Pulling four factors for {year}...")

    df = kp.get_fourfactors(browser, season=str(year))

    filename = DATA / f"kenpom_fourfactors_{year}.csv"
    df.to_csv(filename, index=False)

    print(f"Saved {filename}")