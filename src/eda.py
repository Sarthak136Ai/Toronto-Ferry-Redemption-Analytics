"""
eda.py
──────
Generates all EDA charts and prints summary statistics.
Outputs saved to data/processed/eda_outputs/

Run from project root:
    python src/eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CLEANED_PATH = os.path.join("data", "processed", "ferry_cleaned.csv")
OUT_DIR = os.path.join("data", "processed", "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
COLORS = {"sales": "#2196F3", "redemptions": "#FF5722", "net": "#4CAF50"}


def load():
    df = pd.read_csv(CLEANED_PATH, parse_dates=["timestamp"])
    print(f"[INFO] Loaded: {df.shape}")
    return df


# ── 1. Summary Statistics ─────────────────────────────────────────────────────
def summary_stats(df):
    print("\n===== SUMMARY STATISTICS =====")
    print(df[["sales", "redemptions", "net_movement"]].describe().round(2))
    print(f"\nDate range : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Total sales      : {df['sales'].sum():,}")
    print(f"Total redemptions: {df['redemptions'].sum():,}")
    print(f"Net movement     : {df['net_movement'].sum():,}")
    print(f"Missing values   :\n{df.isnull().sum()}")


# ── 2. Hourly Demand Trends ───────────────────────────────────────────────────
def plot_hourly_demand(df):
    hourly = df.groupby("hour")[["sales", "redemptions"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hourly["hour"] - 0.2, hourly["sales"],       width=0.4, label="Avg Sales",       color=COLORS["sales"])
    ax.bar(hourly["hour"] + 0.2, hourly["redemptions"], width=0.4, label="Avg Redemptions", color=COLORS["redemptions"])
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Tickets per 15-min Interval")
    ax.set_title("Average Hourly Ferry Demand (Sales vs Redemptions)")
    ax.set_xticks(range(0, 24))
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_hourly_demand.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


# ── 3. Daily Demand Trends ────────────────────────────────────────────────────
def plot_daily_demand(df):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily = df.groupby("day_name")[["sales", "redemptions"]].mean().reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(order))
    ax.bar([i - 0.2 for i in x], daily["sales"],       width=0.4, label="Avg Sales",       color=COLORS["sales"])
    ax.bar([i + 0.2 for i in x], daily["redemptions"], width=0.4, label="Avg Redemptions", color=COLORS["redemptions"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(order)
    ax.set_title("Average Daily Ferry Demand by Day of Week")
    ax.set_ylabel("Average Tickets per 15-min Interval")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_daily_demand.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


# ── 4. Seasonal Comparison ────────────────────────────────────────────────────
def plot_seasonal(df):
    order = ["Spring", "Summer", "Fall", "Winter"]
    seasonal = df.groupby("season")[["sales", "redemptions"]].sum().reindex(order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, color in zip(axes, ["sales", "redemptions"], [COLORS["sales"], COLORS["redemptions"]]):
        ax.bar(seasonal["season"], seasonal[col], color=color)
        ax.set_title(f"Total {col.capitalize()} by Season")
        ax.set_ylabel("Total Tickets")
        for i, v in enumerate(seasonal[col]):
            ax.text(i, v + 500, f"{v:,.0f}", ha="center", fontsize=9)
    plt.suptitle("Seasonal Ferry Activity (2015–2025)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "03_seasonal.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


# ── 5. Sales vs Redemptions Distribution ─────────────────────────────────────
def plot_distributions(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, color in zip(axes, ["sales", "redemptions"], [COLORS["sales"], COLORS["redemptions"]]):
        ax.hist(df[col], bins=60, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(df[col].mean(),   color="black", linestyle="--", label=f"Mean={df[col].mean():.1f}")
        ax.axvline(df[col].median(), color="grey",  linestyle=":",  label=f"Median={df[col].median():.1f}")
        ax.set_title(f"Distribution of {col.capitalize()} per 15-min Interval")
        ax.set_xlabel("Count")
        ax.set_ylabel("Frequency")
        ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


# ── 6. Rolling Average Time Series (yearly sample) ───────────────────────────
def plot_rolling_avg(df):
    sample = df[df["year"] == 2019].copy()
    if sample.empty:
        sample = df.copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample["timestamp"], sample["sales_rolling_1h"],
            label="Sales 1-hr Rolling Avg", color=COLORS["sales"], linewidth=1.2)
    ax.plot(sample["timestamp"], sample["redemptions_rolling_1h"],
            label="Redemptions 1-hr Rolling Avg", color=COLORS["redemptions"], linewidth=1.2)
    ax.set_title("1-Hour Rolling Average — Ferry Sales & Redemptions (2019 Sample)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Tickets per Interval")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "05_rolling_avg.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


# ── 7. Net Passenger Movement by Hour ────────────────────────────────────────
def plot_net_movement(df):
    net_hourly = df.groupby("hour")["net_movement"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(net_hourly["hour"], net_hourly["net_movement"],
                  color=[COLORS["net"] if v >= 0 else COLORS["redemptions"] for v in net_hourly["net_movement"]])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Net Passenger Movement (Sales − Redemptions)")
    ax.set_title("Net Passenger Movement by Hour of Day\n(Positive = Net Inflow to Island)")
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "06_net_movement.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


# ── 8. Year-over-Year Monthly Trend ──────────────────────────────────────────
def plot_yoy_trend(df):
    monthly = df.groupby(["year", "month"])["sales"].sum().reset_index()
    monthly["period"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))

    fig, ax = plt.subplots(figsize=(14, 5))
    for year, grp in monthly.groupby("year"):
        ax.plot(grp["month"], grp["sales"], marker="o", label=str(year), linewidth=1.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.set_title("Year-over-Year Monthly Sales Trend (2015–2025)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.legend(title="Year", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "07_yoy_trend.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


# ── 9. Heatmap: Hour × Day of Week ───────────────────────────────────────────
def plot_heatmap(df):
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.pivot_table(values="sales", index="hour", columns="day_name", aggfunc="mean")
    pivot = pivot.reindex(columns=day_order)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, ax=ax, fmt=".1f", annot=False)
    ax.set_title("Average Sales Heatmap: Hour of Day × Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "08_heatmap_hour_day.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


# ── 10. Weekend vs Weekday ────────────────────────────────────────────────────
def plot_weekend_vs_weekday(df):
    grouped = df.groupby("is_weekend")[["sales", "redemptions"]].mean().reset_index()
    grouped["label"] = grouped["is_weekend"].map({0: "Weekday", 1: "Weekend"})

    fig, ax = plt.subplots(figsize=(7, 5))
    x = [0, 1]
    ax.bar([i - 0.2 for i in x], grouped["sales"],       width=0.4, label="Sales",       color=COLORS["sales"])
    ax.bar([i + 0.2 for i in x], grouped["redemptions"], width=0.4, label="Redemptions", color=COLORS["redemptions"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Weekday", "Weekend"])
    ax.set_title("Weekend vs Weekday: Avg Tickets per 15-min Interval")
    ax.set_ylabel("Average Tickets")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "09_weekend_vs_weekday.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


# ── KPI Summary ───────────────────────────────────────────────────────────────
def compute_kpis(df):
    print("\n===== KEY PERFORMANCE INDICATORS =====")
    # Tickets per hour (4 intervals per hour)
    df_hourly = df.groupby(df["timestamp"].dt.floor("h"))[["sales", "redemptions"]].sum()
    print(f"Avg Tickets Sold per Hour     : {df_hourly['sales'].mean():.1f}")
    print(f"Avg Tickets Redeemed per Hour : {df_hourly['redemptions'].mean():.1f}")
    print(f"Peak Demand Hour              : {df.groupby('hour')['sales'].mean().idxmax()}:00")
    print(f"Off-Season (Winter) avg sales : {df[df['season']=='Winter']['sales'].mean():.2f}")
    print(f"Peak Season (Summer) avg sales: {df[df['season']=='Summer']['sales'].mean():.2f}")
    off_season_idx = (df[df["season"]=="Winter"]["sales"].mean() /
                      df[df["season"]=="Summer"]["sales"].mean()) * 100
    print(f"Off-Season Utilization Index  : {off_season_idx:.1f}% of peak season")

    peak_windows = (df[df["is_peak"] == 1]
                    .groupby("hour")["sales"].mean()
                    .sort_values(ascending=False)
                    .head(5))
    print(f"\nTop 5 Peak Demand Hours (avg sales):\n{peak_windows}")


def main():
    df = load()
    summary_stats(df)
    compute_kpis(df)
    plot_hourly_demand(df)
    plot_daily_demand(df)
    plot_seasonal(df)
    plot_distributions(df)
    plot_rolling_avg(df)
    plot_net_movement(df)
    plot_yoy_trend(df)
    plot_heatmap(df)
    plot_weekend_vs_weekday(df)
    print(f"\n[DONE] All EDA charts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
