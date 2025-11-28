import pandas
import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder

weather_cats = ["Clear", "Cloudy", "Foggy", "Rain", "Thunderstorm", "Snow/Ice", "Windy", "Other"]


# Function to do web scrapping
def get_weather_data(loc, date_from, date_to, url, columns):
    # Set up Selenium WebDriver (Make sure chromedriver is in your system path)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(options=options)

    # Define date range (hourly data from 2020 to 2025)
    date_range = pd.date_range(start=date_from, end=date_to, freq="D")
    data_list = []
    for date in tqdm(date_range):
        url_date = url + date.strftime("%Y-%m-%d")

        driver.get(url_date)
        time.sleep(10)  # Allow time for page to load

        try:
            # Wait for the "Daily Observations" table to appear
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//table[contains(@class, 'mat-tabl')]"))
            )

            # Parse the page with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # Locate the correct table
            table = soup.find("table", class_="mat-table")

            if table:
                rows = table.find_all("tr")
                for row in rows[1:]:  # Skip header row
                    cols = row.find_all("td")
                    if len(cols) > 1:
                        hour = cols[0].text.strip()
                        temp = cols[1].text.strip()
                        humidity = cols[3].text.strip()
                        wind = cols[5].text.strip()
                        press = cols[7].text.strip()
                        perc = cols[8].text.strip()
                        cond = cols[9].text.strip()
                        data_list.append([date.strftime("%m/%d/%Y"),
                                             hour, temp, humidity, wind,
                                             press, perc, cond])

        except Exception as e:
            print(f"Crashed. Error on {url}: {e}")
            return pd.DataFrame(data_list, columns=["Date", "Hour"] + columns)

    # Close Selenium browser
    driver.quit()

    # Convert to DataFrame
    df = pd.DataFrame(data_list, columns=["Date", "Hour"] + columns)

    # Print success
    print(f"Data saved for {loc} from {date_from} to {date_to}")
    return df


# Refine data
def refine_weather_data(df, date_from, date_to):
    num_cols = ['Temperature', 'Humidity', 'Wind', 'Pressure', 'Precipitation']
    cat_cols = ['Condition']

    # Remove unwanted texts
    df[num_cols] = remove_unwanted_str(df[num_cols], num_cols)

    # Turn Humidity to %
    df['Humidity'] = df['Humidity'] / 100

    # Convert hour to a range of [0, 23]
    df['Hour'] = convert_to_24_hour(df['Hour'])

    # Aggregate duplicates and fill missing dates with Null
    df = aggregate_duplicates(df, num_cols, cat_cols, date_from, date_to)

    # Fill missing values
    df = fill_missing_hours(df, num_cols, cat_cols)
    return df


def categorize_weather(df):
    # Apply categorization for weather condition
    df["Condition"] = df["Condition"].apply(get_category)
    ava_cats = df["Condition"].unique()

    # OneHotEncoder on weather condition
    encoder = OneHotEncoder()
    encoded_weather = encoder.fit_transform(df[["Condition"]])
    encoded_weather = pd.DataFrame(encoded_weather.toarray(),
                                   columns=encoder.get_feature_names_out(["Condition"]))
    absent_cats = list(set(weather_cats) - set(ava_cats))
    absent_data = pd.DataFrame({f"Condition_{cat}": len(df) * [0.0] for cat in absent_cats})
    encoded_weather = pd.concat((encoded_weather, absent_data), axis=1)

    df.drop(columns=["Condition"], inplace=True)
    df = pd.concat((df, encoded_weather), axis=1)
    return df


# Convert str time of day to a range [0, 23]
def convert_to_24_hour(time_series):

    return pd.to_datetime(time_series, format='%I:%M %p').dt.hour


# Fill missing values
def fill_missing_hours(df, num_cols, cat_cols):
    mean_values_whole = df[num_cols].mean()
    mode_values_whole = df[cat_cols].mode()
    for date, group in df.groupby(['Date'], as_index=False):
        # Identify rows with missing values
        missing_rows = group[group.isna().any(axis=1)]

        # Calculate the mean of all non-NaN values for the existing hours
        mean_values = group[num_cols].mean()
        mode_values = group[cat_cols].mode()

        for idx, row in missing_rows.iterrows():
            # Fill missing values based on the available data
            for col in num_cols:
                if pd.isna(row[col]):  # If value is missing, fill it
                    if pd.isna(mean_values[col]):
                        df.at[idx, col] = mean_values_whole[col]
                    else:
                        df.at[idx, col] = mean_values[col]
            for col in cat_cols:
                if mode_values.empty:
                    df.at[idx, col] = mode_values_whole[col]
                else:
                    df.at[idx, col] = mode_values[col]
    return df


# Aggregate multiple hours
def aggregate_duplicates(df, num_cols, cat_cols, date_from, date_to):
    df['Date'] = pd.to_datetime(df['Date'])

    # Define aggregation functions
    agg_funcs = {col: 'mean' for col in num_cols}
    agg_funcs.update({col: 'last' for col in cat_cols})

    # Apply aggregation for each date-hour pair
    df_aggregated = df.groupby(['Date', 'Hour'], as_index=False).agg(agg_funcs)

    # Generate the full date-hour range from 2015 to 2024
    full_dates = pd.date_range(start=date_from, end=date_to, freq='D')
    full_hours = range(24)  # 0 to 23

    # Create a full DataFrame with all Date-Hour combinations
    full_index = pd.MultiIndex.from_product([full_dates, full_hours], names=['Date', 'Hour'])
    full_df = pd.DataFrame(index=full_index).reset_index()

    # Merge with the aggregated DataFrame to detect missing Date-Hour pairs
    df_complete = full_df.merge(df_aggregated, on=['Date', 'Hour'], how='left')

    return df_complete


# Refine order of hours
def correct_hour_order(df):
    # Sort by Date and hour to ensure correct order
    df_sorted = df.sort_values(by=['Date', 'Hour']).reset_index(drop=True)
    return df_sorted


# Remove unwanted str
def remove_unwanted_str(df, cols):
    for col in cols:
        # Remove any char that is not number \d, period ., or - sign
        df[col] = df[col].replace(r'[^\d.-]', '', regex=True)

        # Convert the cleaned strings to numeric values
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# Define keyword-based categorization for weather condition
def get_category(condition):
    if isinstance(condition, pd.Series):
        condition = condition[0].lower()
    else:
        condition = condition.lower()
    if any(word in condition for word in ["fair"]):
        return "Clear"
    elif any(word in condition for word in ["cloudy", "partly cloudy", "mostly cloudy"]):
        return "Cloudy"
    elif any(word in condition for word in ["fog", "mist", "haze"]):
        return "Foggy"
    elif any(word in condition for word in ["rain", "drizzle", "showers"]):
        return "Rain"
    elif any(word in condition for word in ["thunder", "t-storm", "storm"]):
        return "Thunderstorm"
    elif any(word in condition for word in ["snow", "sleet", "freezing"]):
        return "Snow/Ice"
    elif any(word in condition for word in ["windy", "wdy"]):
        return "Windy"
    else:
        return "Other"
