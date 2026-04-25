# Ferry Capacity Utilization & Operational Efficiency Analytics System
**Toronto Government — Parks, Forestry & Recreation**

---

## Project Overview

This project delivers a pure operational analytics framework for the Toronto Island Ferry system, analyzing 10+ years of ticket sales and redemption data (2015–2025) to identify capacity inefficiencies, congestion patterns, and actionable operational improvements.

**Dataset:** 261,538 records across 15-minute intervals | **Period:** May 2015 – Dec 2025

---

## Project Structure

```
ferry_project/
├── data/
│   └── Toronto_Island_Ferry_Tickets.csv
├── notebooks/
│   └── 01_EDA_and_Feature_Engineering.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   └── kpi_calculator.py
├── dashboard/
│   └── app.py
├── reports/
│   └── research_paper.md
├── requirements.txt
└── README.md
```

---

## Setup Instructions (Windows Terminal)

```bash
# 1. Clone / create project folder
mkdir ferry_project
cd ferry_project

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the CSV in data/ folder
mkdir data
# Copy Toronto_Island_Ferry_Tickets.csv into data/

# 5. Run EDA Notebook
jupyter notebook notebooks/01_EDA_and_Feature_Engineering.ipynb

# 6. Run Streamlit Dashboard
streamlit run dashboard/app.py
```

---

## Key Findings

- **Peak congestion**: 11 AM–3 PM accounts for ~60% of total annual ticket activity
- **COVID-19 impact**: 2020 saw a 71% drop in sales vs 2019; recovery completed by 2022
- **Weekend pressure**: Weekends carry 2.3x the utilization load of weekdays in summer
- **Idle capacity**: ~23% of operating intervals fall below 10% estimated capacity utilization
- **Highest strain year**: 2016 (1.52M tickets sold), rebounded similarly in 2024–2025

---

## Deliverables

1. ✅ EDA + Feature Engineering Notebook
2. ✅ Modular Python source code (data loader, feature engineering, KPIs)
3. ✅ Streamlit Dashboard (multi-tab, filters, KPI cards, heatmaps)
4. ✅ Research Paper (EDA, methodology, insights, recommendations)
