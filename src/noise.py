"""Make the future weather look like a weather forecast instead of the observed weather.

The decoder gets the weather of the hours it forecasts. Our data holds the *observed* weather,
which is better than any real forecast. With future_weather: noisy, forecast-like errors are added:

Numbers (Temperature, Humidity, Wind, Pressure), in their original units:
    error(hour) = daily_error + hourly_error
    daily_error  ~ Normal(0, 0.8 * std)   one value shared by all 24 hours of a forecast
    hourly_error ~ Normal(0, 0.6 * std)   a new value every hour
    so the total error has standard deviation `std` (0.8^2 + 0.6^2 = 1).
    Real forecast errors are mostly a shared offset (e.g. the whole day forecast 3 F too warm),
    which is why the daily part is the larger one.
    Humidity is kept between 0 and 1; Wind is kept at 0 or above.

Precipitation:
    amount * exp(Normal(0, std)), one factor per forecast. Multiplying (instead of adding) keeps
    dry hours dry and avoids negative rain.

Weather condition (Clear, Cloudy, Rain, ...):
    each hour, with probability `condition_change_prob`, the condition is replaced by one drawn
    at random according to how often each condition occurs in the training years.
"""
import numpy as np

from src.features import COLUMNS_TO_SCALE

DAILY_SHARE, HOURLY_SHARE = 0.8, 0.6


def add_weather_noise(future, data, cfg, rng):
    """Return a noisy copy of the future features [windows, hours, features]."""
    noise = cfg["weather_noise"]
    columns = data["future_columns"]
    scaler = data["scaler"]
    future = future.copy()
    n_windows, n_hours, _ = future.shape

    def to_original(col, values):
        if col in COLUMNS_TO_SCALE:
            i = COLUMNS_TO_SCALE.index(col)
            return values * scaler.scale_[i] + scaler.mean_[i]
        return values

    def to_scaled(col, values):
        if col in COLUMNS_TO_SCALE:
            i = COLUMNS_TO_SCALE.index(col)
            return (values - scaler.mean_[i]) / scaler.scale_[i]
        return values

    # Numbers: daily offset + hourly wiggle
    for col in ["Temperature", "Humidity", "Wind", "Pressure"]:
        j = columns.index(col)
        std = noise[col]
        daily = rng.normal(0, DAILY_SHARE * std, size=(n_windows, 1))
        hourly = rng.normal(0, HOURLY_SHARE * std, size=(n_windows, n_hours))
        values = to_original(col, future[:, :, j]) + daily + hourly
        if col == "Humidity":
            values = np.clip(values, 0, 1)
        if col == "Wind":
            values = np.clip(values, 0, None)
        future[:, :, j] = to_scaled(col, values)

    # Precipitation: multiply by a random factor
    j = columns.index("Precipitation")
    factor = np.exp(rng.normal(0, noise["Precipitation"], size=(n_windows, 1)))
    future[:, :, j] = to_scaled("Precipitation", to_original("Precipitation", future[:, :, j]) * factor)

    # Weather condition: sometimes replace by a random condition (by how common it is)
    condition_cols = [i for i, c in enumerate(columns) if c.startswith("Condition_")]
    train_rows = data["df"]["split"] == "train"
    frequency = data["df"].loc[train_rows, [columns[i] for i in condition_cols]].mean().to_numpy()
    change = rng.random((n_windows, n_hours)) < noise["condition_change_prob"]
    new_condition = rng.choice(len(condition_cols), size=change.sum(), p=frequency / frequency.sum())
    onehot = np.zeros((change.sum(), len(condition_cols)), dtype=future.dtype)
    onehot[np.arange(change.sum()), new_condition] = 1
    block = future[:, :, condition_cols]
    block[change] = onehot
    future[:, :, condition_cols] = block
    return future
