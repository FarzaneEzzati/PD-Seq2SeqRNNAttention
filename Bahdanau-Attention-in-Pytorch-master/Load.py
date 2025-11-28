import pandas as pd
from TimeFeatures import build_time_features


def get_demand(date_from, date_to):
    # Trun to datetime
    date_from = pd.to_datetime(date_from)
    date_to = pd.to_datetime(date_to)

    # Get year first
    year_from = date_from.year
    year_to = date_to.year

    l_df = []
    for year in range(year_from, year_to + 1):

        d = pd.read_excel(f'Data/Native_Load/Native_Load_{year}.xlsx', engine='openpyxl')

        time_col = d.columns[0]  # First column name

        # Convert the time column to string first to handle both formats
        d[time_col] = d[time_col].astype(str)

        has_24h = d[time_col].str.contains('24:00', na=False).any()
        d[time_col] = d[time_col].str.replace(r'24:', '00:', regex=True)

        # Remove unwanted text like 'DST' and other non-date characters
        d[time_col] = d[time_col].str.replace(r'DST', '', regex=False)  # Remove DST text

        # Convert to datetime, using 'coerce' to handle errors
        d[time_col] = pd.to_datetime(d[time_col], errors='coerce', infer_datetime_format=False)
        d[time_col] = d[time_col].dt.round('H')

        # For rows where the hour is 00 (i.e., originally 24), we add one day
        if has_24h:
            d[time_col] = d[time_col].apply(lambda x: x + pd.Timedelta(days=1) if x.hour == 0 and x.minute == 0 else x)

        # Group by the 'DateHour' and calculate the mean of all numerical columns
        d = d.groupby('HourEnding').agg('mean').reset_index()

        # Optional: Ensure time is in a consistent format (e.g., YYYY-MM-DD HH:MM)
        d[time_col] = d[time_col].dt.strftime('%m/%d/%Y %H:%M')
        l_df.append(d)
    # Concat all years
    l_df = pd.concat(l_df, ignore_index=True)

    # Get available date range
    first_date = l_df['HourEnding'].values[0]
    last_date = l_df['HourEnding'].values[-1]

    # Fill missing values
    l_df = fill_missing_dates(l_df, 'HourEnding', first_date, last_date)

    # Pick the requested range
    l_df = l_df[(l_df['HourEnding'] >= pd.to_datetime(date_from).floor('D')) &
            (l_df['HourEnding'] < pd.to_datetime(date_to) + pd.Timedelta(days=1))]

    return l_df


# Fill missing value cells
def fill_missing_dates(df, time_col, date_from, date_to):
    df[time_col] = pd.to_datetime(df[time_col])

    # Create the full range of dates at hourly intervals
    full_range = pd.date_range(start=pd.to_datetime(date_from).floor('D'),
                               end=date_to,
                               freq='H')


    # Create a DataFrame with the full date range
    full_df = pd.DataFrame(full_range, columns=[time_col])

    # Merge the full range with your DataFrame, using the time column
    merged_df = pd.merge(full_df, df, how='left', on=time_col)

    # Fill missing rows with the mean of corresponding columns (excluding the time column)
    for col in merged_df.columns:
        if col != time_col:
            merged_df[col] = merged_df[col].fillna(merged_df[col].mean())

    return merged_df


# Scale demand
def scale_demand(df, downscale_rate, num_houses, region):
    df['Demand'] = 1000 * (1 - downscale_rate) * df[region] / num_houses
    df.drop(columns=region, inplace=True)
    return df


# Get time features
def get_time_features(df):
    df = pd.concat((df, build_time_features(df[['Date', 'Hour']])), axis=1)
    df.drop(columns=['Date', 'Hour'], inplace=True)
    return df