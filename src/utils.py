import sys

import pandas as pd
import yaml


def config_path():
    """config.yaml, or another settings file given when running a step: python 3_train.py other.yaml"""
    return sys.argv[1] if len(sys.argv) > 1 else "config.yaml"


def load_config():
    """Read the settings file and add the settings of the chosen location under cfg["loc"]."""
    with open(config_path(), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["loc"] = cfg["locations"][cfg["location"]]
    return cfg


def fill_short_gaps(df, max_gap_hours, text_columns=()):
    """Make the data hourly without holes.

    Short gaps (up to max_gap_hours) are filled: numbers by interpolation, text by the last value.
    Longer gaps are left out, because guessing days of missing data would be misleading.
    """
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h", name="timestamp")
    df = df.reindex(full_range)

    # Find each run of missing hours and its length
    missing = df.isna().any(axis=1)
    run_id = (missing != missing.shift()).cumsum()
    run_length = missing.groupby(run_id).transform("size")
    long_gap = missing & (run_length > max_gap_hours)

    number_columns = [c for c in df.columns if c not in text_columns]
    df[number_columns] = df[number_columns].interpolate(method="time", limit_area="inside")
    for col in text_columns:
        df[col] = df[col].ffill()

    df = df[~long_gap].dropna()
    print(f"    filled {missing.sum() - long_gap.sum()} missing hours, left out {long_gap.sum()} hours in long gaps")
    return df
