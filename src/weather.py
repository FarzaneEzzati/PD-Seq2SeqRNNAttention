"""Hourly weather from Wunderground: scrape missing days and clean the data.

The raw file data/raw/weather/<location>.csv keeps the observations as scraped:
Date, Hour, Temperature (F), Humidity, Wind (mph), Pressure (in), Precipitation (in), Condition
"""
import os
import time

import pandas as pd

from src.utils import fill_short_gaps

RAW_COLUMNS = ["Date", "Hour", "Temperature", "Humidity", "Wind", "Pressure", "Precipitation", "Condition"]
NUMBER_COLUMNS = ["Temperature", "Humidity", "Wind", "Pressure", "Precipitation"]
CONDITIONS = ["Clear", "Cloudy", "Foggy", "Rain", "Thunderstorm", "Snow/Ice", "Windy", "Other"]

# Readings outside these ranges are scraping errors (e.g. 0 inHg pressure) and are treated as missing
VALID_RANGES = {"Temperature": (0.1, 125), "Humidity": (0.01, 1), "Wind": (0, 75),
                "Pressure": (20, 32), "Precipitation": (0, 10)}


# ---------------------------------------------------------------- scraping
def missing_days(raw_file, start_date, end_date):
    """Days between start_date and end_date that have no observations in the raw file."""
    all_days = pd.date_range(start_date, end_date, freq="D")
    if not os.path.exists(raw_file):
        return list(all_days)
    have = set(pd.to_datetime(pd.read_csv(raw_file, usecols=["Date"])["Date"], format="%m/%d/%Y"))
    return [day for day in all_days if day not in have]


def scrape_missing_days(url, raw_file, start_date, end_date):
    """Scrape only the missing days and add them to the raw file, saving after every month.

    Needs: pip install selenium beautifulsoup4, and Google Chrome.
    If it stops halfway, run it again: it continues with the days still missing.
    """
    from selenium import webdriver

    days = missing_days(raw_file, start_date, end_date)
    print(f"    {len(days)} days to scrape")
    if not days:
        return

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    try:
        months = pd.Series(days).groupby(pd.Series(days).dt.to_period("M"))
        for month, month_days in months:
            rows = []
            for day in month_days:
                rows += scrape_one_day(driver, url, day)
            if rows:
                new = pd.DataFrame(rows, columns=RAW_COLUMNS)
                new.to_csv(raw_file, mode="a", header=not os.path.exists(raw_file), index=False)
            print(f"    {month}: {len(rows)} observations saved")
    finally:
        driver.quit()


def scrape_one_day(driver, url, day):
    from bs4 import BeautifulSoup
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions
    from selenium.webdriver.support.ui import WebDriverWait

    driver.get(url + day.strftime("%Y-%m-%d"))
    try:
        # Wait until the observations table has loaded
        WebDriverWait(driver, 30).until(expected_conditions.presence_of_element_located(
            (By.XPATH, "//table[contains(@class, 'mat-table')]//td")))
    except Exception:
        print(f"    {day.date()}: page did not load, will retry next time")
        return []
    time.sleep(1)

    table = BeautifulSoup(driver.page_source, "html.parser").find("table", class_="mat-table")
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.text.strip() for td in tr.find_all("td")]
        if len(cells) > 9:
            rows.append([day.strftime("%m/%d/%Y"), cells[0], cells[1], cells[3], cells[5],
                         cells[7], cells[8], cells[9]])
    return rows


# ---------------------------------------------------------------- cleaning
def clean_weather(raw, max_gap_hours=24):
    """Turn the raw observations into one row per hour."""
    df = raw.copy()
    date = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    hour = pd.to_datetime(df["Hour"].str.strip(), format="%I:%M %p").dt.hour
    df["timestamp"] = date + pd.to_timedelta(hour, unit="h")

    for col in NUMBER_COLUMNS:
        # Remove units such as '56 °F' and keep the number
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce")
    df.loc[df["Humidity"] > 1, "Humidity"] /= 100  # some scrapes give %, others a fraction
    for col, (low, high) in VALID_RANGES.items():
        df.loc[(df[col] < low) | (df[col] > high), col] = None

    df["Condition"] = df["Condition"].apply(condition_group)

    # Several observations in the same hour: average the numbers, keep the last condition
    rules = {col: "mean" for col in NUMBER_COLUMNS}
    rules["Condition"] = "last"
    df = df.groupby("timestamp").agg(rules)
    print(f"    weather: {df.index.min()} to {df.index.max()}")
    return fill_short_gaps(df, max_gap_hours, text_columns=["Condition"])


def condition_group(condition):
    """Put the many Wunderground condition texts into a few groups."""
    if not isinstance(condition, str):
        return "Other"
    c = condition.lower()
    if "fair" in c or "clear" in c:
        return "Clear"
    if "cloudy" in c:
        return "Cloudy"
    if any(word in c for word in ["fog", "mist", "haze"]):
        return "Foggy"
    if any(word in c for word in ["rain", "drizzle", "showers"]):
        return "Rain"
    if any(word in c for word in ["thunder", "t-storm", "storm"]):
        return "Thunderstorm"
    if any(word in c for word in ["snow", "sleet", "freezing"]):
        return "Snow/Ice"
    if any(word in c for word in ["windy", "wdy"]):
        return "Windy"
    return "Other"
