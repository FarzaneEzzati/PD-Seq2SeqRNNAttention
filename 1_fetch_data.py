"""Step 1: make sure the raw data is available.

With fetch_from_internet: false (default) nothing is downloaded; it only checks data/raw.
With fetch_from_internet: true it downloads missing ERCOT years and scrapes missing weather days.
"""
import glob

import pandas as pd

from src import ercot, weather
from src.utils import load_config

cfg = load_config()
start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
weather_file = f"data/raw/weather/{cfg['location']}.csv"

if cfg["fetch_from_internet"]:
    print("Downloading missing ERCOT years...")
    ercot.download_missing_years(start.year, end.year)
    print("Scraping missing weather days (slow: about one page per day)...")
    weather.scrape_missing_days(cfg["loc"]["weather_url"], weather_file, start, end)

# Report what is available
years = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob("data/raw/ercot/Native_Load_*.xls*"))
missing_years = [y for y in range(start.year, end.year + 1) if y not in years]
print(f"ERCOT years available: {years}")
if missing_years:
    print(f"WARNING: missing ERCOT years {missing_years}. Set fetch_from_internet: true to download them.")

missing = weather.missing_days(weather_file, start, end)
print(f"Weather ({cfg['location']}): {len(missing)} days without data between {start.date()} and {end.date()}")
