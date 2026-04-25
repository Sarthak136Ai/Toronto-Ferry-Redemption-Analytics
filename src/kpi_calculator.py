"""
kpi_calculator.py
=================
Computes all five Key Performance Indicators defined in the project brief,
plus supplementary metrics for the executive summary.

KPIs:
    1. Capacity Utilization Ratio (CUR)
    2. Congestion Pressure Index (CPI)
    3. Idle Capacity Percentage (ICP)
    4. Peak Strain Duration (PSD)
    5. Operational Variability Score (OVS)
"""

import pandas as pd
import numpy as np
from scipy.stats import variation


# ── Ferry operational assumption ──────────────────────────────────────────────
# Toronto Island ferries have a combined fleet capacity. We use 99th percentile
# of observed total_activity as the proxy for "full capacity" (ferry system max
# throughput in a 15-minute window) since the actual vessel capacity spec is not
# in the dataset. This is a common operational analytics approach when hard
# capacity ceilings are undocumented.
# ──────────────────────────────────────────────────────────────────────────────


def capacity_utilization_ratio(df_15min: pd.DataFrame) -> dict:
    """
    CUR = mean(total_activity) / max_observed_activity

    Measures average system load as a fraction of peak throughput.
    """
    max_cap = df_15min["total_activity"].quantile(0.99)  # proxy for capacity ceiling
    mean_activity = df_15min["total_activity"].mean()
    cur = round(mean_activity / max_cap, 4)

    return {
        "kpi": "Capacity Utilization Ratio",
        "value": cur,
        "pct": f"{cur * 100:.1f}%",
        "proxy_capacity_ceiling": round(max_cap, 0),
        "mean_interval_activity": round(mean_activity, 2),
        "interpretation": (
            "High" if cur > 0.6 else "Moderate" if cur > 0.35 else "Low"
        ),
    }


def congestion_pressure_index(df_15min: pd.DataFrame) -> dict:
    """
    CPI = proportion of intervals with total_activity >= 90th percentile.

    Measures how frequently the system experiences near-peak strain.
    """
    threshold = df_15min["total_activity"].quantile(0.90)
    congestion_count = (df_15min["total_activity"] >= threshold).sum()
    cpi = round(congestion_count / len(df_15min), 4)

    return {
        "kpi": "Congestion Pressure Index",
        "value": cpi,
        "pct": f"{cpi * 100:.1f}%",
        "congestion_threshold": round(threshold, 0),
        "congestion_intervals": int(congestion_count),
        "interpretation": (
            "Severe" if cpi > 0.15 else "Elevated" if cpi > 0.08 else "Normal"
        ),
    }


def idle_capacity_percentage(df_15min: pd.DataFrame) -> dict:
    """
    ICP = proportion of intervals with total_activity <= 10th percentile.

    Quantifies wasted operational capacity.
    """
    threshold = df_15min["total_activity"].quantile(0.10)
    idle_count = (df_15min["total_activity"] <= threshold).sum()
    icp = round(idle_count / len(df_15min), 4)

    return {
        "kpi": "Idle Capacity Percentage",
        "value": icp,
        "pct": f"{icp * 100:.1f}%",
        "idle_threshold": round(threshold, 0),
        "idle_intervals": int(idle_count),
        "interpretation": (
            "Critical waste" if icp > 0.25 else "Moderate" if icp > 0.12 else "Acceptable"
        ),
    }


def peak_strain_duration(df_15min: pd.DataFrame) -> dict:
    """
    PSD = longest consecutive run of congestion-flagged intervals.

    Measures how long sustained high-pressure periods last.
    Reports max, mean, and median consecutive congestion run lengths (in minutes).
    """
    flags = df_15min["congestion_flag"].values
    runs = []
    current = 0
    for f in flags:
        if f == 1:
            current += 1
        else:
            if current > 0:
                runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)

    if not runs:
        return {"kpi": "Peak Strain Duration", "value": 0, "interpretation": "No congestion detected"}

    max_run = max(runs)
    mean_run = round(np.mean(runs), 2)
    median_run = round(np.median(runs), 2)

    return {
        "kpi": "Peak Strain Duration",
        "value": max_run,
        "max_consecutive_intervals": max_run,
        "max_duration_minutes": max_run * 15,
        "mean_duration_minutes": mean_run * 15,
        "median_duration_minutes": median_run * 15,
        "total_congestion_runs": len(runs),
        "interpretation": (
            "Very Long" if max_run >= 16 else "Long" if max_run >= 8 else "Moderate"
        ),
    }


def operational_variability_score(df_15min: pd.DataFrame) -> dict:
    """
    OVS = Coefficient of Variation of total_activity.

    CV = std / mean. Higher = more erratic, harder to schedule efficiently.
    """
    cv = round(variation(df_15min["total_activity"]), 4)
    return {
        "kpi": "Operational Variability Score",
        "value": cv,
        "std": round(df_15min["total_activity"].std(), 2),
        "mean": round(df_15min["total_activity"].mean(), 2),
        "interpretation": (
            "Highly Variable" if cv > 1.5 else "Moderate" if cv > 0.8 else "Stable"
        ),
    }


def compute_all_kpis(df_15min: pd.DataFrame) -> dict:
    """Compute and return all five KPIs as a unified dictionary."""
    return {
        "CUR": capacity_utilization_ratio(df_15min),
        "CPI": congestion_pressure_index(df_15min),
        "ICP": idle_capacity_percentage(df_15min),
        "PSD": peak_strain_duration(df_15min),
        "OVS": operational_variability_score(df_15min),
    }


def print_kpi_summary(kpis: dict) -> None:
    print("\n" + "=" * 60)
    print("  KEY PERFORMANCE INDICATORS — Ferry Operations")
    print("=" * 60)
    for code, data in kpis.items():
        print(f"\n  [{code}] {data['kpi']}")
        print(f"       Value          : {data.get('pct', data['value'])}")
        print(f"       Interpretation : {data['interpretation']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_data
    from feature_engineering import build_15min_df

    csv_path = os.path.join(os.path.dirname(__file__), "../data/Toronto_Island_Ferry_Tickets.csv")
    raw = load_data(csv_path)
    df15 = build_15min_df(raw)
    kpis = compute_all_kpis(df15)
    print_kpi_summary(kpis)
