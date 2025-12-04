# CareCast – Pediatric ICU Bed Utilization Forecasting

CareCast is an interactive forecasting app that helps hospital operations teams anticipate **Pediatric ICU (PICU) bed utilization** over the next 7–60 days.  
It combines **time-series modeling**, **scenario analysis**, and a lightweight **Streamlit** UI so clinicians and planners can quickly stress-test capacity under different demand shocks.

---

## 🔍 Key Features

- **End-to-end time-series pipeline**
  - SARIMAX models for short- and medium-term PICU bed utilization forecasts
  - Automated data cleaning, resampling, and outlier handling
  - Rolling retrain-ready structure for future extension

- **Interactive Streamlit app**
  - KPI cards for current occupancy, forecasted utilization, and risk flags
  - Sparklines and trend charts for historical vs forecasted demand
  - Sliders to simulate demand/capacity shocks (e.g., +20% admissions, −10% staffed beds)
  - State-level PICU availability and utilization overview

- **Production-friendly structure**
  - `requirements.txt` for reproducible environments
  - Clean separation of **data**, **notebooks**, and **app code**
  - MIT-licensed for reuse and extension

---

## 🗂 Project Structure

```bash
carecast_hospital_forecasting/
├── data/                     # (ignored by Git) raw & processed datasets live here locally
├── notebooks/
│   └── 01_explore_and_clean.ipynb   # EDA, cleaning, and feature engineering
├── src/                      # (optional) reusable utilities / modeling helpers
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # Project documentation