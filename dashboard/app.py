"""
app.py — Ferry Capacity Utilization & Operational Efficiency Dashboard
======================================================================
Multi-tab Streamlit dashboard for Toronto Island Ferry operations analytics.

Tabs:
    1. 📊 KPI Overview
    2. 📈 Capacity Utilization Timeline
    3. 🔥 Congestion & Idle Heatmaps
    4. 🌿 Seasonal Efficiency Comparison
    5. 📋 Raw Data Explorer

Run:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, '..', 'src')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data/raw', 'Toronto_Island_Ferry_Tickets.csv')
sys.path.insert(0, SRC_DIR)

from data_loader import load_data
from feature_engineering import build_15min_df, build_hourly_df, build_daily_df
from kpi_calculator import compute_all_kpis

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ferry Ops Analytics — Toronto",
    page_icon="⛴️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .kpi-title { color: #a8b2d8; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { color: #e94560; font-size: 32px; font-weight: 700; margin: 8px 0; }
    .kpi-sub   { color: #8892b0; font-size: 12px; }
    .kpi-badge-green  { background: #0d4f3c; color: #4caf50; padding: 4px 10px; border-radius: 20px; font-size: 11px; }
    .kpi-badge-yellow { background: #4a3f0d; color: #ffc107; padding: 4px 10px; border-radius: 20px; font-size: 11px; }
    .kpi-badge-red    { background: #4f0d0d; color: #f44336; padding: 4px 10px; border-radius: 20px; font-size: 11px; }
    .section-header { border-left: 4px solid #e94560; padding-left: 12px; margin: 20px 0 10px 0; }
    div[data-testid="stMetricValue"] { font-size: 28px !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and processing ferry data...")
def load_all_data():
    raw   = load_data(DATA_PATH)
    df15  = build_15min_df(raw)
    dfh   = build_hourly_df(df15)
    dfd   = build_daily_df(df15)
    kpis  = compute_all_kpis(df15)
    return df15, dfh, dfd, kpis

df15, dfh, dfd, kpis = load_all_data()

SEASON_ORDER = ['Spring', 'Summer', 'Fall', 'Winter']
MONTH_NAMES  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛴️ Ferry Analytics")
    st.markdown("**Toronto Island Ferry Terminal**")
    st.divider()

    st.markdown("### 🎛️ Global Filters")

    years = sorted(df15['year'].unique())
    selected_years = st.multiselect(
        "Select Year(s)", years, default=years, key="years"
    )

    seasons = ['All'] + SEASON_ORDER
    selected_season = st.selectbox("Season", seasons, index=0)

    day_type = st.radio("Day Type", ["All", "Weekdays Only", "Weekends Only"])

    st.divider()
    st.markdown("### ⚡ Threshold Alerts")
    congestion_thresh = st.slider(
        "Congestion OLI Alert Threshold", 0.50, 1.00, 0.85, step=0.05
    )
    idle_thresh = st.slider(
        "Idle OLI Alert Threshold", 0.00, 0.30, 0.05, step=0.01
    )

    st.divider()
    st.markdown("### 📐 Granularity")
    granularity = st.radio("Chart Resolution", ["15-Minute", "Hourly", "Daily"])

    st.divider()
    st.caption("Data: 2015–2025 | 261,538 Records")


# ── Apply sidebar filters to 15-min df ──────────────────────────────────────
def apply_filters(df):
    df = df[df['year'].isin(selected_years)]
    if selected_season != 'All':
        df = df[df['season'] == selected_season]
    if day_type == "Weekdays Only":
        df = df[df['is_weekend'] == 0]
    elif day_type == "Weekends Only":
        df = df[df['is_weekend'] == 1]
    return df

df_filtered = apply_filters(df15)
dfd_filtered = apply_filters(dfd.assign(
    year=pd.to_datetime(dfd['date']).dt.year,
    season=pd.to_datetime(dfd['date']).dt.month.map({
        12:'Winter',1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',
        6:'Summer',7:'Summer',8:'Summer',9:'Fall',10:'Fall',11:'Fall'
    }),
    is_weekend=pd.to_datetime(dfd['date']).dt.dayofweek.isin([5,6]).astype(int)
))

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPI Overview",
    "📈 Utilization Timeline",
    "🔥 Congestion & Idle Heatmaps",
    "🌿 Seasonal Efficiency",
    "📋 Data Explorer"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — KPI OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<h2 class="section-header">Key Performance Indicators</h2>', unsafe_allow_html=True)
    st.markdown(f"Filtered data: **{len(df_filtered):,} intervals** | Years: {selected_years}")

    # Recompute KPIs on filtered data
    from kpi_calculator import compute_all_kpis as _kpis
    filtered_kpis = _kpis(df_filtered) if len(df_filtered) > 100 else kpis

    def badge(interp):
        positive_words = ['Acceptable', 'Stable', 'Normal', 'Low']
        negative_words = ['Critical', 'Severe', 'Highly Variable', 'Very Long']
        if any(w in interp for w in positive_words):
            return f'<span class="kpi-badge-green">✓ {interp}</span>'
        elif any(w in interp for w in negative_words):
            return f'<span class="kpi-badge-red">⚠ {interp}</span>'
        else:
            return f'<span class="kpi-badge-yellow">~ {interp}</span>'

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        cur = filtered_kpis['CUR']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Capacity Utilization</div>
            <div class="kpi-value">{cur['pct']}</div>
            <div class="kpi-sub">CUR</div>
            {badge(cur['interpretation'])}
        </div>""", unsafe_allow_html=True)

    with col2:
        cpi = filtered_kpis['CPI']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Congestion Index</div>
            <div class="kpi-value">{cpi['pct']}</div>
            <div class="kpi-sub">CPI — % intervals congested</div>
            {badge(cpi['interpretation'])}
        </div>""", unsafe_allow_html=True)

    with col3:
        icp = filtered_kpis['ICP']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Idle Capacity</div>
            <div class="kpi-value">{icp['pct']}</div>
            <div class="kpi-sub">ICP — % intervals idle</div>
            {badge(icp['interpretation'])}
        </div>""", unsafe_allow_html=True)

    with col4:
        psd = filtered_kpis['PSD']
        val = f"{psd.get('max_duration_minutes', 0)} min"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Peak Strain Duration</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">Longest congestion run</div>
            {badge(psd['interpretation'])}
        </div>""", unsafe_allow_html=True)

    with col5:
        ovs = filtered_kpis['OVS']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Variability Score</div>
            <div class="kpi-value">{ovs['value']:.2f}</div>
            <div class="kpi-sub">OVS — Coeff. of Variation</div>
            {badge(ovs['interpretation'])}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Activity breakdown by time band ──────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Activity by Time Band")
        band_stats = df_filtered.groupby('time_band').agg(
            avg_activity=('total_activity', 'mean'),
            total_intervals=('total_activity', 'count')
        ).reset_index()
        band_order = ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
        band_stats = band_stats.set_index('time_band').reindex(band_order).dropna().reset_index()
        fig = px.bar(band_stats, x='time_band', y='avg_activity',
                     color='avg_activity', color_continuous_scale='Reds',
                     labels={'time_band': 'Time Band', 'avg_activity': 'Avg Activity'},
                     title='Average Interval Activity by Time Band')
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Yearly Total Activity")
        yr_stats = df_filtered.groupby('year')['total_activity'].sum().reset_index()
        fig = px.bar(yr_stats, x='year', y='total_activity',
                     color='total_activity', color_continuous_scale='Blues',
                     labels={'year': 'Year', 'total_activity': 'Total Activity'},
                     title='Total Annual Ticket Activity')
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Alert zones ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚡ Threshold-Based Efficiency Alerts")

    congestion_count = (df_filtered['oli'] >= congestion_thresh).sum()
    idle_count = (df_filtered['oli'] <= idle_thresh).sum()
    total = len(df_filtered)

    a1, a2, a3 = st.columns(3)
    a1.metric("🔴 Congestion Alert Intervals",
              f"{congestion_count:,}",
              f"{congestion_count/total*100:.1f}% of filtered intervals")
    a2.metric("🟡 Idle Alert Intervals",
              f"{idle_count:,}",
              f"{idle_count/total*100:.1f}% of filtered intervals")
    a3.metric("🟢 Normal Intervals",
              f"{total - congestion_count - idle_count:,}",
              f"{(total - congestion_count - idle_count)/total*100:.1f}% of filtered intervals")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UTILIZATION TIMELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<h2 class="section-header">Capacity Utilization Timeline</h2>', unsafe_allow_html=True)

    if granularity == "15-Minute":
        # Show up to 2000 points for performance
        plot_df = df_filtered.tail(2000)[['Timestamp', 'total_activity', 'oli', 'congestion_flag', 'idle_flag']].copy()
        x_col = 'Timestamp'
    elif granularity == "Hourly":
        plot_df = dfh[dfh['year'].isin(selected_years)].copy()
        if selected_season != 'All':
            plot_df = plot_df[plot_df['season'] == selected_season]
        x_col = 'hour_bucket'
    else:
        plot_df = dfd_filtered.copy()
        plot_df['date'] = pd.to_datetime(plot_df['date'])
        x_col = 'date'

    # Main timeline chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3],
                        subplot_titles=['Total Activity (Ticket Count)', 'Operational Load Index (OLI)'])

    fig.add_trace(go.Scatter(
        x=plot_df[x_col], y=plot_df['total_activity'],
        mode='lines', name='Total Activity',
        line=dict(color='#4fc3f7', width=1),
        fill='tozeroy', fillcolor='rgba(79,195,247,0.1)'
    ), row=1, col=1)

    if 'oli' in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df[x_col], y=plot_df['oli'],
            mode='lines', name='OLI',
            line=dict(color='#ff6b6b', width=1.5)
        ), row=2, col=1)

        # Congestion threshold line
        fig.add_hline(y=congestion_thresh, line_dash='dash',
                      line_color='red', opacity=0.7,
                      annotation_text=f'Congestion ({congestion_thresh})',
                      row=2, col=1)
        fig.add_hline(y=idle_thresh, line_dash='dash',
                      line_color='gray', opacity=0.7,
                      annotation_text=f'Idle ({idle_thresh})',
                      row=2, col=1)

    fig.update_layout(height=550, showlegend=True,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'))
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')

    st.plotly_chart(fig, use_container_width=True)

    # Hourly average pattern
    st.markdown("#### Average Activity Profile by Hour of Day")
    hourly_pattern = df_filtered.groupby('hour').agg(
        avg_sales=('sales', 'mean'),
        avg_redemptions=('redemptions', 'mean'),
        avg_activity=('total_activity', 'mean')
    ).reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hourly_pattern['hour'], y=hourly_pattern['avg_sales'],
                              name='Avg Sales', mode='lines+markers',
                              line=dict(color='#4fc3f7', width=2.5)))
    fig2.add_trace(go.Scatter(x=hourly_pattern['hour'], y=hourly_pattern['avg_redemptions'],
                              name='Avg Redemptions', mode='lines+markers',
                              line=dict(color='#ff8a65', width=2.5)))
    fig2.add_trace(go.Scatter(x=hourly_pattern['hour'], y=hourly_pattern['avg_activity'],
                              name='Total Activity', mode='lines',
                              line=dict(color='#a5d6a7', width=2, dash='dot'),
                              fill='tozeroy', fillcolor='rgba(165,214,167,0.05)'))
    fig2.add_vrect(x0=11, x1=15, fillcolor='red', opacity=0.05,
                   annotation_text='Peak Zone', annotation_position='top left')
    fig2.update_layout(height=380, xaxis_title='Hour of Day',
                       yaxis_title='Avg Tickets per 15-min Interval',
                       paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(color='white'))
    fig2.update_xaxes(tickvals=list(range(0, 24)), gridcolor='rgba(255,255,255,0.05)')
    fig2.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONGESTION & IDLE HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<h2 class="section-header">Congestion & Idle Period Heatmaps</h2>', unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("#### OLI Heatmap: Hour × Month")
        pivot_oli = df_filtered.groupby(['hour', 'month'])['oli'].mean().unstack()
        pivot_oli.columns = [MONTH_NAMES[i-1] for i in pivot_oli.columns]

        fig = px.imshow(pivot_oli, color_continuous_scale='YlOrRd',
                        labels=dict(x='Month', y='Hour', color='Avg OLI'),
                        title='Operational Load Index by Hour and Month',
                        aspect='auto')
        fig.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    with col_h2:
        st.markdown("#### Congestion Frequency: Hour × Day of Week")
        dow_order_num = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_cong = df_filtered.groupby(['hour', 'day_of_week'])['congestion_flag'].mean().unstack()
        pivot_cong = pivot_cong.reindex(columns=dow_order_num)

        fig = px.imshow(pivot_cong * 100, color_continuous_scale='Reds',
                        labels=dict(x='Day', y='Hour', color='% Congested'),
                        title='Congestion Frequency (%) by Hour and Day of Week',
                        aspect='auto')
        fig.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col_h3, col_h4 = st.columns(2)

    with col_h3:
        st.markdown("#### Idle Capacity Heatmap: Hour × Month")
        pivot_idle = df_filtered.groupby(['hour', 'month'])['idle_flag'].mean().unstack() * 100
        pivot_idle.columns = [MONTH_NAMES[i-1] for i in pivot_idle.columns]

        fig = px.imshow(pivot_idle, color_continuous_scale='Blues_r',
                        labels=dict(x='Month', y='Hour', color='% Idle'),
                        title='Idle Capacity Frequency (%) by Hour and Month',
                        aspect='auto')
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

    with col_h4:
        st.markdown("#### Redemption Pressure: Hour × Season")
        pivot_rp = df_filtered.groupby(['hour', 'season'])['redemption_pressure'].mean().unstack()
        pivot_rp = pivot_rp.reindex(columns=SEASON_ORDER)

        fig = px.imshow(pivot_rp, color_continuous_scale='PuBuGn',
                        labels=dict(x='Season', y='Hour', color='Redemption Pressure'),
                        title='Redemption Pressure Ratio by Hour and Season',
                        aspect='auto')
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SEASONAL EFFICIENCY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<h2 class="section-header">Seasonal Efficiency Comparison</h2>', unsafe_allow_html=True)

    # Season summary metrics
    season_summary = df_filtered.groupby('season').agg(
        avg_activity=('total_activity', 'mean'),
        avg_oli=('oli', 'mean'),
        congestion_pct=('congestion_flag', 'mean'),
        idle_pct=('idle_flag', 'mean'),
        total_intervals=('total_activity', 'count')
    ).reindex(SEASON_ORDER).reset_index()
    season_summary['congestion_pct'] = (season_summary['congestion_pct'] * 100).round(1)
    season_summary['idle_pct'] = (season_summary['idle_pct'] * 100).round(1)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        fig = px.bar(season_summary, x='season', y='avg_oli',
                     color='season', title='Average OLI by Season',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='white'), xaxis_title='Season',
                          yaxis_title='Average OLI')
        st.plotly_chart(fig, use_container_width=True)

    with col_s2:
        fig = px.bar(season_summary, x='season',
                     y=['congestion_pct', 'idle_pct'],
                     barmode='group', title='Congestion vs Idle % by Season',
                     labels={'value': '% of Intervals', 'variable': 'Category'},
                     color_discrete_map={'congestion_pct': '#e94560', 'idle_pct': '#4fc3f7'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        newnames = {'congestion_pct': 'Congestion %', 'idle_pct': 'Idle %'}
        fig.for_each_trace(lambda t: t.update(name=newnames[t.name]))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Weekday vs Weekend by season
    st.markdown("#### Weekday vs Weekend OLI — By Season")
    wk_season = df_filtered.groupby(['season', 'is_weekend'])['oli'].mean().reset_index()
    wk_season['type'] = wk_season['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
    wk_season = wk_season[wk_season['season'].isin(SEASON_ORDER)]

    fig = px.bar(wk_season, x='season', y='oli', color='type', barmode='group',
                 category_orders={'season': SEASON_ORDER},
                 title='Weekday vs Weekend Operational Load Index by Season',
                 color_discrete_map={'Weekday': '#4fc3f7', 'Weekend': '#ff6b6b'})
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                      xaxis_title='Season', yaxis_title='Avg OLI')
    st.plotly_chart(fig, use_container_width=True)

    # Yearly trend by season
    st.markdown("#### Annual Activity Trend by Season")
    yr_season = df_filtered.groupby(['year', 'season'])['total_activity'].sum().reset_index()
    yr_season = yr_season[yr_season['season'].isin(SEASON_ORDER)]

    fig = px.line(yr_season, x='year', y='total_activity', color='season',
                  markers=True, title='Total Annual Activity by Season (2015–2025)',
                  category_orders={'season': SEASON_ORDER},
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                      xaxis_title='Year', yaxis_title='Total Activity')
    st.plotly_chart(fig, use_container_width=True)

    # Styled table
    st.markdown("#### Season Summary Table")
    display_df = season_summary[['season', 'avg_activity', 'avg_oli', 'congestion_pct', 'idle_pct']].copy()
    display_df.columns = ['Season', 'Avg Activity', 'Avg OLI', 'Congestion %', 'Idle %']
    display_df['Avg Activity'] = display_df['Avg Activity'].round(1)
    display_df['Avg OLI'] = display_df['Avg OLI'].round(4)
    st.dataframe(display_df.style.background_gradient(subset=['Avg OLI', 'Congestion %'],
                                                       cmap='YlOrRd').format(precision=2),
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<h2 class="section-header">Raw Data Explorer</h2>', unsafe_allow_html=True)

    st.markdown(f"**Showing filtered data: {len(df_filtered):,} rows**")

    cols_to_show = st.multiselect(
        "Select columns to display",
        options=df_filtered.columns.tolist(),
        default=['Timestamp', 'sales', 'redemptions', 'total_activity',
                 'oli', 'season', 'time_band', 'is_weekend',
                 'congestion_flag', 'idle_flag']
    )

    n_rows = st.slider("Number of rows to display", 50, 5000, 500, step=50)
    st.dataframe(
        df_filtered[cols_to_show].tail(n_rows).reset_index(drop=True),
        use_container_width=True,
        height=400
    )

    st.markdown("---")
    st.markdown("#### Download Filtered Data")
    csv_export = df_filtered[cols_to_show].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_export,
        file_name="ferry_filtered_data.csv",
        mime='text/csv'
    )

    st.markdown("---")
    st.markdown("#### Statistical Summary of Filtered Data")
    st.dataframe(
        df_filtered[['sales', 'redemptions', 'total_activity', 'oli', 'redemption_pressure']].describe().round(4),
        use_container_width=True
    )
