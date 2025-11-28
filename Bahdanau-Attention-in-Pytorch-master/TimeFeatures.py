import pandas as pd
import numpy as np
import math
import warnings
from sklearn.preprocessing import StandardScaler
import argparse


def encode_day_of_year(day_of_year, year):
    # Calculate the day of the year
    max_days_year = 366 if is_leap_year(year) else 365
    sine = np.sin(2 * np.pi * (day_of_year-1) / max_days_year)
    cosine = np.cos(2 * np.pi * (day_of_year-1) / max_days_year)
    return sine, cosine


def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def encode_month_of_year(month_of_year):
    max_months = 12
    sine = np.sin(2 * np.pi * (month_of_year-1) / max_months)
    cosine = np.cos(2 * np.pi * (month_of_year-1) / max_months)
    return sine, cosine


def encode_day_of_week(day_of_week):
    max_days = 7
    sine = np.sin(2 * np.pi * day_of_week / max_days)
    cosine = np.cos(2 * np.pi * day_of_week / max_days)
    return sine, cosine


def encode_hour_of_day(hour_of_day):
    max_hours = 24
    sine = np.sin(2 * np.pi * hour_of_day / max_hours)
    cosine = np.cos(2 * np.pi * hour_of_day / max_hours)
    return sine, cosine


def encode_week_of_year(week_of_year):
    max_weeks = 365/7
    sine = np.sin(2 * np.pi * (week_of_year - 1) / max_weeks)
    cosine = np.cos(2 * np.pi * (week_of_year - 1) / max_weeks)
    return sine, cosine


def build_time_features(time_data):
    time_data['Date'] = pd.to_datetime(time_data['Date'], format='%m/%d/%Y')
    time_data['Year'] = time_data['Date'].dt.year
    time_data['Month_of_Year'] = time_data['Date'].dt.month
    time_data['Day_of_Year'] = time_data['Date'].dt.dayofyear
    time_data['Hour_of_Day'] = time_data['Hour']
    time_data['Week_of_Year'] = time_data['Date'].dt.weekofyear
    time_data['Day_of_Week'] = time_data['Date'].dt.weekday  # start from 0
    time_data['Weekend'] = time_data['Date'].dt.weekday.isin([5, 6]).astype(int)

    data_built = pd.DataFrame()
    data_built['Weekend'] = time_data['Weekend']
    data_built['sine_moy'], data_built['cosine_moy'] = encode_month_of_year(time_data['Month_of_Year'])
    data_built['sine_doy'], data_built['cosine_doy'] = encode_day_of_year(time_data['Day_of_Year'],
                                                                        time_data['Year'].iloc[0])
    data_built['sine_dow'], data_built['cosine_dow'] = encode_day_of_week(time_data['Day_of_Week'])
    data_built['sine_hod'], data_built['cosine_hod'] = encode_hour_of_day(time_data['Hour_of_Day'])
    data_built['sine_woy'], data_built['cine_woy'] = encode_week_of_year(time_data['Week_of_Year'])

    # columns: [month_of_year, demand, weekend, sine_moy, cosine_moy, sine_doy, cosine_doy,
    #           sine_dow, cosine_dow, sine_hod, cosine_hod, sine_woy, cosine_woy]
    return data_built



