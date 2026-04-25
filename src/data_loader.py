"""
data_loader.py
==============
Handles data ingestion, type casting, sorting, and basic quality checks
for the Toronto Island Ferry Tickets dataset.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw CSV, parse timestamps, and sort chronologically.

    Parameters
    ----------
    filepath : str
        Path to Toronto_Island_Ferry_Tickets.csv

    Returns
    -------
    pd.DataFrame
        Cleaned, sorted DataFrame ready for feature engineering.
    """
    df = pd.read_csv(filepath)

    # Parse timestamps
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Sort ascending by time
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Rename columns for convenience
    df.rename(
        columns={
            "Sales Count": "sales",
            "Redemption Count": "redemptions",
        },
        inplace=True,
    )

    return df


def run_quality_checks(df: pd.DataFrame) -> dict:
    """
    Run data quality diagnostics and return a summary dictionary.

    Checks:
    - Missing values per column
    - Negative counts (operational anomaly)
    - Zero-activity intervals
    - Duplicate timestamps
    - Temporal gaps (intervals where 15-min gap is broken)
    """
    results = {}

    results["total_rows"] = len(df)
    results["missing_values"] = df.isnull().sum().to_dict()
    results["negative_sales"] = int((df["sales"] < 0).sum())
    results["negative_redemptions"] = int((df["redemptions"] < 0).sum())
    results["zero_activity_rows"] = int(
        ((df["sales"] == 0) & (df["redemptions"] == 0)).sum()
    )
    results["duplicate_timestamps"] = int(df["Timestamp"].duplicated().sum())

    # Check for unexpected time gaps (should be 15-min intervals)
    time_diffs = df["Timestamp"].diff().dropna()
    expected_gap = pd.Timedelta("15min")
    gaps = time_diffs[time_diffs > expected_gap * 2]
    results["large_temporal_gaps"] = len(gaps)
    results["date_range"] = {
        "start": str(df["Timestamp"].min()),
        "end": str(df["Timestamp"].max()),
    }

    return results


def print_quality_report(df: pd.DataFrame) -> None:
    """Pretty-print the quality check results."""
    report = run_quality_checks(df)
    print("=" * 55)
    print("  DATA QUALITY REPORT — Toronto Island Ferry")
    print("=" * 55)
    print(f"  Total Rows         : {report['total_rows']:,}")
    print(f"  Date Range         : {report['date_range']['start'][:10]} → {report['date_range']['end'][:10]}")
    print(f"  Missing Values     : {report['missing_values']}")
    print(f"  Negative Sales     : {report['negative_sales']}")
    print(f"  Negative Redemptions: {report['negative_redemptions']}")
    print(f"  Zero-Activity Rows : {report['zero_activity_rows']:,}")
    print(f"  Duplicate Timestamps: {report['duplicate_timestamps']}")
    print(f"  Large Temporal Gaps: {report['large_temporal_gaps']}")
    print("=" * 55)


if __name__ == "__main__":
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "../data/Toronto_Island_Ferry_Tickets.csv")
    df = load_data(csv_path)
    print_quality_report(df)
    print(df.head())
