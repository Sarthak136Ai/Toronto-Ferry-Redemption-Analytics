"""
feature_engineering.py
=======================
Derives all analytical features from the cleaned 15-minute interval data.
Produces 15-min, hourly, and daily aggregations with KPI columns.

Features engineered:
    - total_activity      : sales + redemptions
    - redemption_pressure : redemptions / (sales + 1)
    - oli                 : Operational Load Index (normalized total_activity)
    - idle_flag           : 1 if interval is in bottom 10th percentile of activity
    - hour, day_of_week, month, year, season, time_band
    - is_weekend
    - rolling_1h_avg      : 4-period rolling mean of total_activity (smoothing)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall",   10: "Fall",  11: "Fall",
}

TIME_BAND_MAP = {
    range(5, 9): "Early Morning",
    range(9, 12): "Morning",
    range(12, 17): "Afternoon",
    range(17, 21): "Evening",
}


def get_time_band(hour: int) -> str:
    for rng, label in TIME_BAND_MAP.items():
        if hour in rng:
            return label
    return "Night"


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based columns."""
    df = df.copy()
    df["hour"]        = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.day_name()
    df["day_num"]     = df["Timestamp"].dt.dayofweek      # 0=Mon, 6=Sun
    df["month"]       = df["Timestamp"].dt.month
    df["year"]        = df["Timestamp"].dt.year
    df["date"]        = df["Timestamp"].dt.date
    df["is_weekend"]  = df["day_num"].isin([5, 6]).astype(int)
    df["season"]      = df["month"].map(SEASON_MAP)
    df["time_band"]   = df["hour"].apply(get_time_band)
    return df


def add_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived operational load features."""
    df = df.copy()

    # Total activity load
    df["total_activity"] = df["sales"] + df["redemptions"]

    # Redemption Pressure Ratio: are ferries being used more than sold?
    df["redemption_pressure"] = df["redemptions"] / (df["sales"] + 1)

    # Rolling smoothing over 4 intervals (= 1 hour)
    df["rolling_1h_avg"] = (
        df["total_activity"]
        .rolling(window=4, min_periods=1)
        .mean()
        .round(2)
    )

    # Operational Load Index — normalized total activity [0, 1]
    scaler = MinMaxScaler()
    df["oli"] = scaler.fit_transform(df[["total_activity"]])
    df["oli"] = df["oli"].round(4)

    # Idle flag: bottom 10th percentile of total activity
    threshold = df["total_activity"].quantile(0.10)
    df["idle_flag"] = (df["total_activity"] <= threshold).astype(int)

    # Congestion flag: top 10th percentile
    congestion_threshold = df["total_activity"].quantile(0.90)
    df["congestion_flag"] = (df["total_activity"] >= congestion_threshold).astype(int)

    return df


def build_15min_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return the fully-featured 15-minute interval DataFrame."""
    df = add_temporal_features(df)
    df = add_activity_features(df)
    return df


def build_hourly_df(df_15min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to hourly resolution."""
    df = df_15min.copy()
    df["hour_bucket"] = df["Timestamp"].dt.floor("h")

    agg = df.groupby("hour_bucket").agg(
        sales=("sales", "sum"),
        redemptions=("redemptions", "sum"),
        total_activity=("total_activity", "sum"),
        redemption_pressure=("redemption_pressure", "mean"),
        oli=("oli", "mean"),
        idle_intervals=("idle_flag", "sum"),
        congestion_intervals=("congestion_flag", "sum"),
    ).reset_index()

    agg["hour"]      = agg["hour_bucket"].dt.hour
    agg["year"]      = agg["hour_bucket"].dt.year
    agg["month"]     = agg["hour_bucket"].dt.month
    agg["season"]    = agg["month"].map(SEASON_MAP)
    agg["is_weekend"] = agg["hour_bucket"].dt.dayofweek.isin([5, 6]).astype(int)
    agg["time_band"] = agg["hour"].apply(get_time_band)

    return agg


def build_daily_df(df_15min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily resolution."""
    df = df_15min.copy()

    agg = df.groupby("date").agg(
        sales=("sales", "sum"),
        redemptions=("redemptions", "sum"),
        total_activity=("total_activity", "sum"),
        avg_oli=("oli", "mean"),
        idle_intervals=("idle_flag", "sum"),
        congestion_intervals=("congestion_flag", "sum"),
        peak_activity=("total_activity", "max"),
    ).reset_index()

    agg["date"]      = pd.to_datetime(agg["date"])
    agg["year"]      = agg["date"].dt.year
    agg["month"]     = agg["date"].dt.month
    agg["season"]    = agg["month"].map(SEASON_MAP)
    agg["is_weekend"] = agg["date"].dt.dayofweek.isin([5, 6]).astype(int)
    agg["day_of_week"] = agg["date"].dt.day_name()

    # Idle capacity percentage per day (out of 96 possible 15-min intervals)
    agg["idle_pct"] = (agg["idle_intervals"] / 96 * 100).round(2)

    return agg


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_data

    csv_path = os.path.join(os.path.dirname(__file__), "../data/Toronto_Island_Ferry_Tickets.csv")
    raw = load_data(csv_path)
    df15 = build_15min_df(raw)
    dfh  = build_hourly_df(df15)
    dfd  = build_daily_df(df15)

    print("15-min shape:", df15.shape)
    print("Hourly shape:", dfh.shape)
    print("Daily shape :", dfd.shape)
    print(df15.head(3))
