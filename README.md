### 🚀 Live Demo  
👉 **CareCast Streamlit App:**  
https://carecasthospitalforecasting-8qmkkseyqqjzirmttukwvw.streamlit.app/

🏥 CareCast — Pediatric ICU Capacity Forecasting System

CareCast is a data-driven forecasting system designed to analyze historical hospital utilization trends and predict Pediatric ICU (PICU) bed demand across the United States.
The system provides 7–60 day capacity forecasts, interactive visual dashboards, and anomaly-aware time-series insights for operational planning.

⸻

🚀 Key Features

📊 Forecasting Engine
	•	SARIMAX-based time-series model
	•	Handles missing values, anomalies, and irregular reporting
	•	Generates short-term and medium-term PICU utilization forecasts

🗺️ Interactive Streamlit Dashboard
	•	KPI cards (current occupancy, forecast range, % change)
	•	State-level capacity maps
	•	Trend visualizations (historical & predicted)
	•	CSV upload option for custom datasets

🛠️ Automated Data Pipeline
	•	Data cleaning (outlier removal, NA imputation, smoothing)
	•	Dataset versioning
	•	Support for multiple CSV inputs

📁 Modular Code Structure
	•	src/ contains forecasting logic, preprocessing utilities, and plotting functions
	•	notebooks/ contains exploratory analysis and model development notebooks
	•	Root directory includes production-ready Streamlit app

    
### 📁 Project Structure

```text
carecast_hospital_forecasting/
├── data/
│   ├── raw/                # Original datasets (ignored in .gitignore)
│   └── processed/          # Cleaned datasets
├── notebooks/
│   └── 01_explore_and_clean.ipynb
├── src/
│   ├── preprocess.py       # Cleaning, anomaly handling
│   ├── forecast.py         # SARIMAX forecasting engine
│   └── visualize.py        # Plotting and KPI helper functions
├── streamlit_app.py        # Main dashboard application
├── requirements.txt        # Python dependencies
├── LICENSE
└── README.md

📥 Datasets

This project uses publicly available HHS datasets, including:
	•	Hospital Utilization (State-Level Time Series)
	•	Facility-Level Capacity Data
	•	Contains fields such as staffed beds, occupied ICU beds, pediatric availability, etc.

Large datasets are not stored in the repo due to GitHub’s 100MB limit.
Users may place their own CSVs inside data/raw/.

⸻

🧠 Modeling Approach

Cleaning & Preprocessing
	•	Forward/backward fill for missing values
	•	Rolling mean smoothing for noisy series
	•	Outlier clipping based on IQR thresholds
	•	Weekly aggregation to stabilize reporting cycles

Forecasting
	•	Seasonal ARIMA (SARIMAX)
	•	Trend + seasonal + exogenous signal support
	•	Automatic order selection during experimentation

Outputs
	•	Forecasted ICU utilization
	•	Confidence intervals
	•	Anomaly flags
	•	KPI summaries

🖥️ How to Run Locally

1️⃣ Clone the repository
git clone https://github.com/VishakShashikumar/PICU_CareCast_Hospital_Forecasting.git
cd PICU_CareCast_Hospital_Forecasting
2️⃣ Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Add your datasets
Place your CSV files into:
data/raw/
Expected filenames:
hospital_utilization_state_timeseries.csv
hospital_capacity.csv
5️⃣ Run the Streamlit App
streamlit run streamlit_app.py

The UI will appear at:

👉 http://localhost:8501


🌟 Future Enhancements
	•	LSTM / Prophet model comparison
	•	State-by-state model auto-selection
	•	Real-time API ingestion
	•	Automated weekly retraining pipeline
	•	Cloud deployment (AWS / Streamlit Cloud)

⸻

📜 License

This project is released under the MIT License, enabling full use for academic, research, and organizational purposes.
