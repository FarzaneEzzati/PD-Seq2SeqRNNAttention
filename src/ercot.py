"""ERCOT hourly load: download the yearly files and read them.

Files come from https://www.ercot.com/gridinfo/load/load_hist and are saved as
data/raw/ercot/Native_Load_<year>.xlsx (older years are .xls).
"""
import glob
import io
import os
import re
import zipfile

import pandas as pd
import requests

from src.utils import fill_short_gaps

ERCOT_PAGE = "https://www.ercot.com/gridinfo/load/load_hist"
REGIONS = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST", "ERCOT"]
HEADERS = {"User-Agent": "Mozilla/5.0"}


def find_download_links():
    """Return {year: link} for every yearly load file listed on the ERCOT page."""
    html = requests.get(ERCOT_PAGE, headers=HEADERS, timeout=60).text
    links = {}
    for link, year in re.findall(r'href="([^"]*native_load_(\d{4})[^"]*\.(?:zip|xlsx?))"', html, re.IGNORECASE):
        links.setdefault(int(year), link)  # the newest upload is listed first
    return links


def download_missing_years(first_year, last_year, folder="data/raw/ercot"):
    """Download the years that are not in the folder yet."""
    os.makedirs(folder, exist_ok=True)
    links = find_download_links()
    for year in range(first_year, last_year + 1):
        if glob.glob(f"{folder}/Native_Load_{year}.xls*"):
            continue
        if year not in links:
            print(f"    {year}: not available on the ERCOT page")
            continue
        print(f"    {year}: downloading")
        content = requests.get(links[year], headers=HEADERS, timeout=120).content
        if links[year].lower().endswith(".zip"):
            # The zip holds one Excel file
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                name = [n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls"))][0]
                content = z.read(name)
        extension = ".xls" if links[year].lower().endswith(".xls") else ".xlsx"
        with open(f"{folder}/Native_Load_{year}{extension}", "wb") as f:
            f.write(content)


def parse_hour_ending(column):
    """Convert ERCOT 'HourEnding' values to the start of each hour.

    ERCOT writes '01/01/2017 01:00' for the hour 00:00-01:00, uses '24:00' for midnight,
    and marks the repeated autumn hour with 'DST'. The 2015-2016 files hold real dates.
    """
    if pd.api.types.is_datetime64_any_dtype(column):
        times = column
    else:
        text = column.astype(str).str.replace("DST", "").str.strip()
        is_24 = text.str.contains(" 24:00")
        times = pd.to_datetime(text.str.replace(" 24:00", " 00:00"), format="mixed")
        times = times.where(~is_24, times + pd.Timedelta(days=1))
    return times.dt.round("h") - pd.Timedelta(hours=1)


def read_load(folder="data/raw/ercot", max_gap_hours=24):
    """Read all yearly files into one hourly table (MW per region)."""
    tables = []
    for file in sorted(glob.glob(f"{folder}/Native_Load_*.xls*")):
        df = pd.read_excel(file)
        df = df.rename(columns={df.columns[0]: "timestamp"})  # 'HourEnding' or 'Hour Ending'
        df["timestamp"] = parse_hour_ending(df["timestamp"])
        tables.append(df[["timestamp"] + [r for r in REGIONS if r in df.columns]])
    if not tables:
        raise FileNotFoundError(f"No ERCOT files in {folder}. Set fetch_from_internet: true and run 1_fetch_data.py")

    load = pd.concat(tables).groupby("timestamp").mean()  # averages duplicate hours (DST, year overlaps)
    print(f"    ERCOT load: {load.index.min()} to {load.index.max()}")
    return fill_short_gaps(load, max_gap_hours)
