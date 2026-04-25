"""
preprocess.py
─────────────
Loads raw ferry ticket CSV, cleans it, engineers features,
and writes cleaned data to data/processed/ferry_cleaned.csv

Run from project root:
    python src/preprocess.py
"""

import os
import pandas as pd
import numpy as np

RAW_PATH = os.path.join("data", "raw", "Toronto_Island_Ferry_Tickets.csv")
PROCESSED_DIR = os.path.join("data", "processed")
OUT_PATH = os.path.join(PROCESSED_DIR, "ferry_cleaned.csv")


def load_data(path: str) -> pd.DataFrame:
    print(f"[INFO] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Cleaning data...")

    # Rename columns for consistency
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "_id": "id",
        "Timestamp": "timestamp",
        "Redemption Count": "redemptions",
        "Sales Count": "sales"
    })

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows with unparseable timestamps
    before = len(df)
    df = df.dropna(subset=["timestamp"])
    after = len(df)
    print(f"[INFO] Dropped {before - after} rows with bad timestamps.")

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Clamp negative counts to 0 (data integrity)
    for col in ["sales", "redemptions"]:
        neg = (df[col] < 0).sum()
        if neg:
            print(f"[WARN] {neg} negative values in '{col}' — clamping to 0.")
        df[col] = df[col].clip(lower=0)

    # Outlier detection using IQR (cap, not drop — operationally plausible spikes exist)
    for col in ["sales", "redemptions"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_fence = Q3 + 3 * IQR
        outliers = (df[col] > upper_fence).sum()
        print(f"[INFO] '{col}': {outliers} extreme outliers above {upper_fence:.0f} (kept, flagged).")
        df[f"{col}_outlier"] = df[col] > upper_fence

    # Fill any remaining NaN in counts with 0
    df["sales"] = df["sales"].fillna(0).astype(int)
    df["redemptions"] = df["redemptions"].fillna(0).astype(int)

    print(f"[INFO] After cleaning: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Engineering features...")

    ts = df["timestamp"]

    df["year"]        = ts.dt.year
    df["month"]       = ts.dt.month
    df["day"]         = ts.dt.day
    df["hour"]        = ts.dt.hour
    df["minute"]      = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek          # 0=Mon … 6=Sun
    df["day_name"]    = ts.dt.day_name()
    df["week"]        = ts.dt.isocalendar().week.astype(int)

    # Weekend flag
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Season (Northern Hemisphere)
    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall",   10: "Fall",  11: "Fall"}
    df["season"] = df["month"].map(season_map)

    # Time-of-day bucket (ferry operational labels)
    def time_bucket(hour):
        if 6 <= hour < 10:
            return "Morning Rush"
        elif 10 <= hour < 14:
            return "Midday"
        elif 14 <= hour < 18:
            return "Afternoon"
        elif 18 <= hour < 22:
            return "Evening"
        else:
            return "Off-Hours"

    df["time_bucket"] = df["hour"].apply(time_bucket)

    # Core KPI: Net Passenger Movement
    # Positive = more people arriving on island (net inflow)
    # Negative = more people leaving island (net outflow)
    df["net_movement"] = df["sales"] - df["redemptions"]

    # Cumulative sum (running balance — think of island crowd estimate)
    df["cumulative_net"] = df["net_movement"].cumsum()

    # Rolling hourly averages (4 intervals × 15 min = 1 hour)
    df["sales_rolling_1h"]       = df["sales"].rolling(4, min_periods=1).mean()
    df["redemptions_rolling_1h"] = df["redemptions"].rolling(4, min_periods=1).mean()

    # Rolling 4-hour averages (16 intervals)
    df["sales_rolling_4h"]       = df["sales"].rolling(16, min_periods=1).mean()
    df["redemptions_rolling_4h"] = df["redemptions"].rolling(16, min_periods=1).mean()

    # Peak flag: hour > 75th percentile of hourly sales
    hourly_avg = df.groupby("hour")["sales"].transform("mean")
    peak_threshold = hourly_avg.quantile(0.75)
    df["is_peak"] = (hourly_avg >= peak_threshold).astype(int)

    print(f"[INFO] Features engineered. Final shape: {df.shape}")
    return df


def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Cleaned data saved to: {path}")


def main():
    df = load_data(RAW_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    save_data(df, OUT_PATH)
    print("\n[DONE] Preprocessing complete.")
    print(df.dtypes)
    print(df.head(3))


if __name__ == "__main__":
    main()
