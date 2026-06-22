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


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

browser = login(EMAIL, PASSWORD)

for year in range(2021, 2026):

    try:
        print(f"Pulling {year}...")
        df = kp.get_efficiency(browser, season=str(year))
        filename = DATA_DIR / f"kenpom_{year}.csv"
        df.to_csv(filename, index=False)
        print(f"✓ Saved {filename}")

    except Exception as e:
        print(f"✗ Failed for {year}")
        print(e)