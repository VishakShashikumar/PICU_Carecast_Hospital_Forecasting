import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# ---------- OPENAI CLIENT ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None
def get_carebot_response(
    user_question: str,
    state: str,
    latest_util: float,
    scenario_util: float,
    horizon_days: int,
    demand_shock: float,
    capacity_shock: float,
) -> str:
    """
    Returns a response from the LLM if OPENAI_API_KEY is set,
    otherwise falls back to a deterministic, safe answer.
    """

    # Fallback if no key configured
    if not OPENAI_API_KEY:
        return (
            "The CareBot LLM is not configured (missing OPENAI_API_KEY). "
            "However, from an operations perspective you might consider:\n\n"
            "• Monitoring utilization trends daily and setting internal alert thresholds\n"
            "• Pre-identifying surge spaces and cross-training staff\n"
            "• Coordinating with nearby facilities for potential transfers\n"
            "• Reviewing elective procedure schedules during forecasted peaks\n"
            "• Ensuring clear communication plans with clinical leadership and regional partners\n"
            "\nFor clinical decisions or emergencies, always contact licensed clinicians or emergency services."
        )

        # Build context for the model
    system_msg = f"""
You are CareBot, an AI capacity-planning assistant for pediatric ICU (PICU) beds in US hospitals.

You DO:
- Focus on hospital operations, bed capacity, staffing, and surge planning.
- Use the numeric context I give you: current utilization %, scenario utilization %, shocks, and horizon.
- Speak clearly and concisely.

Output format (IMPORTANT):
- Organize your answer into 3–5 sections.
- Each section MUST start with a Markdown heading like:
  ## Risk level summary
  ## Surge planning and triggers
  ## Staffing and resource actions
  ## Coordination and communication
- Inside each section, use short bullet points, NOT long paragraphs.

You DO NOT:
- Give medical diagnosis, treatment recommendations, or triage individual patients.
- Tell people what medication, dose, or procedure to use.
- Replace clinicians or emergency services.

If the user sounds like they describe an emergency or individual patient, remind them:
"Contact clinical staff or emergency services. I can only help with planning and operational scenarios."

Frame your answers for the state: {state}.
Current observed PICU utilization is about {latest_util:.1f}%.
Scenario forecast utilization over the next {horizon_days} days is about {scenario_util:.1f}%,
given a demand shock of {demand_shock:+.1f}% and capacity change of {capacity_shock:+.1f}%.
    """.strip()
    
    user_msg = f"""
User question: {user_question}
State: {state}
Current utilization: {latest_util:.1f}%
Scenario utilization: {scenario_util:.1f}%
Demand shock: {demand_shock:+.1f}% | Capacity shock: {capacity_shock:+.1f}%
    """.strip()

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_output_tokens=450,
        )

        text = response.output[0].content[0].text
        return text

    except Exception as e:
        # If API fails, fall back gracefully
        return (
            "CareBot ran into a technical issue while contacting the AI service. "
            "Here are some general planning steps you might still consider:\n\n"
            "• Review current PICU occupancy and trends over the next few days\n"
            "• Escalate to surge plans if utilization remains high\n"
            "• Coordinate with nearby hospitals on transfer protocols\n"
            "• Adjust elective admissions if capacity remains constrained\n\n"
            f"(Technical detail: {e})"
        )
    # END of get_carebot_response()

def render_carebot_answer(answer: str) -> None:
    """
    Split CareBot's markdown answer into sections starting with '## '
    and show each section inside a collapsible Streamlit expander.
    """
    lines = answer.splitlines()
    sections = []
    current_title = "CareBot Recommendations"
    current_body = []

    for line in lines:
        if line.startswith("## "):
            if current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_title, "\n".join(current_body).strip()))

    for title, body in sections:
        if not body:
            continue
        with st.expander(title):
            st.markdown(body)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="CareCast – Pediatric ICU Bed Forecasting",
    layout="wide"
)

# ---------- GLOBAL THEME (LIGHT / DARK) ----------
def apply_global_theme(dark: bool):
    # Light vs dark palette
    if dark:
        bg = "#050816"
        card_bg = "#111827"
        text = "#f9fafb"
        subtext = "#9ca3af"
        border = "#1f2937"
    else:
        bg = "#f5f6f9"
        card_bg = "#ffffff"
        text = "#111827"
        subtext = "#6b7280"
        border = "#e5e7eb"

    st.markdown(
        f"""
        <style>
        /* Global font + background */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: {bg};
            color: {text};
        }}

        .main {{
            background-color: {bg};
        }}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }}

        /* KPI chips */
        .kpi-chip {{
            background: {card_bg};
            border-radius: 16px;
            padding: 12px 14px;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.18);
            border: 1px solid {border};
        }}
        .kpi-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {subtext};
        }}
        .kpi-value {{
            font-size: 1.35rem;
            font-weight: 600;
            color: {text};
            margin-top: 2px;
        }}
        .kpi-sub {{
            font-size: 0.70rem;
            color: {subtext};
            margin-top: 2px;
        }}

        /* badges stay same colors, etc… */
        </style>
        """,
        unsafe_allow_html=True,
    )
# ---------- OPENAI CLIENT ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None


# ---------- LOAD & PREP DATA ----------
@st.cache_data
def load_state_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df[
        [
            "state",
            "date",
            "staffed_pediatric_icu_bed_occupancy",
            "total_staffed_pediatric_icu_beds",
        ]
    ]

    df["staffed_pediatric_icu_bed_occupancy"] = pd.to_numeric(
        df["staffed_pediatric_icu_bed_occupancy"], errors="coerce"
    )
    df["total_staffed_pediatric_icu_beds"] = pd.to_numeric(
        df["total_staffed_pediatric_icu_beds"], errors="coerce"
    )

    df = df[df["total_staffed_pediatric_icu_beds"] > 0]
    df = df.dropna()

    df["picu_utilization_pct"] = (
        df["staffed_pediatric_icu_bed_occupancy"]
        / df["total_staffed_pediatric_icu_beds"]
        * 100.0
    )

    df = df.sort_values(["state", "date"])

    return df


@st.cache_data
def load_facility_data(csv_path: str) -> pd.DataFrame:
    """
    Facility-level dataset. Column names may differ depending on version.
    This implementation expects at least:
      - 'state'
      - 'hospital_name'
      - 'collection_week' (or 'date')
      - 'latitude' / 'lat'
      - 'longitude' / 'lon'
      - some capacity/occupancy columns
    Adjust column names below if needed.
    """
    df = pd.read_csv(csv_path)

    # Try to normalize date column
    if "collection_week" in df.columns:
        df["collection_week"] = pd.to_datetime(df["collection_week"], errors="coerce")
        df.rename(columns={"collection_week": "date"}, inplace=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    # Try to guess lat/lon columns
    lat_col = None
    lon_col = None
    for c in df.columns:
        lc = c.lower()
        if lat_col is None and ("lat" == lc or "latitude" in lc):
            lat_col = c
        if lon_col is None and ("lon" == lc or "lng" in lc or "longitude" in lc):
            lon_col = c

    df["lat"] = df[lat_col] if lat_col in df.columns else np.nan
    df["lon"] = df[lon_col] if lon_col in df.columns else np.nan

    # Pick some capacity columns if they exist – adjust as needed
    cap_cols = [c for c in df.columns if "icu_beds" in c.lower() or "beds" in c.lower()]
    if cap_cols:
        df["facility_capacity_proxy"] = pd.to_numeric(df[cap_cols[0]], errors="coerce")
    else:
        df["facility_capacity_proxy"] = np.nan

    return df


DATA_FILE_STATE = "hospital_utilization_state_timeseries.csv"
DATA_FILE_FACILITY = "hospital_facility_capacity.csv"

try:
    full_df = load_state_data(DATA_FILE_STATE)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_FILE_STATE}` in this folder. "
        "Rename your state-level CSV to that name, or update DATA_FILE_STATE in streamlit_app.py."
    )
    st.stop()

facility_df = None
try:
    facility_df = load_facility_data(DATA_FILE_FACILITY)
except FileNotFoundError:
    st.warning(
        f"Facility-level file `{DATA_FILE_FACILITY}` not found. "
        "Hospital-level map will be disabled until this file is added."
    )

# --------- SIDEBAR CONTROLS ---------
with st.sidebar:
    # Dark mode toggle
    dark_mode = st.checkbox("🌗 Dark mode", value=False, key="dark_mode")
    
    # Top title with icon
    st.markdown("### ⚙️ Forecast & Scenario Settings")

    st.markdown(
        """
        <div style="font-size:0.8rem; margin-bottom:0.75rem; color:#6b7280;">
          Tune the <b>state</b>, <b>horizon</b>, and <b>what-if shocks</b> to see how PICU utilization might change.
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# Apply theme based on sidebar toggle
apply_global_theme(st.session_state.get("dark_mode", False))

# ========= MAIN PAGE HEADER ==========
st.markdown(
    """
    <h1 style='margin-bottom:0px;'>🩺 CareCast – Pediatric ICU Bed Utilization Forecast</h1>
    <p style='font-size:16px; color:gray; margin-top:4px;'>
        Interactive hospital capacity intelligence built on HHS utilization data.<br>
        Includes SARIMAX forecasting, scenario analysis, maps, and an AI assistant.<br>
        <b>Not for clinical decision-making or emergency use.</b>
    </p>
    <hr style='margin-top:10px; margin-bottom:20px;'>
    """,
    unsafe_allow_html=True,
)
# --- State & horizon ---
st.markdown("#### 🗺️ State & horizon")
state_list = sorted(full_df["state"].unique())
selected_state = st.selectbox(
    "Select state",
     state_list,
)

forecast_days = st.slider(
    "Forecast horizon (days)",
    min_value=7,
    max_value=60,
    value=30,
)

st.markdown("---")

# --- Scenario shocks ---
st.markdown("#### 📊 Scenario: Demand & capacity shocks")
# Little legend chips with icons
st.markdown(
        """
        <div style="font-size:0.75rem; margin-bottom:0.35rem;">
          <span style="
            padding:2px 8px;
            border-radius:999px;
            background:#e0edff;
            margin-right:6px;
            ">
            📈 demand
          </span>
          <span style="
            padding:2px 8px;
            border-radius:999px;
            background:#e4f8f0;
            ">
            🛏 capacity
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

demand_shock_pct = st.slider(
        "Change in demand (admissions) %",
        min_value=-50,
        max_value=100,
        value=0,
        step=5,
    )

capacity_change_pct = st.slider(
        "Change in staffed PICU bed capacity %",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
    )

st.caption("🧪 Ops planning only – not for clinical decisions.")
state_df = full_df[full_df["state"] == selected_state].copy()
# ----------BUILD TIME SERIES FOR THIS STATE ----------
df_ts = state_df[["date", "picu_utilization_pct"]].rename(
    columns={"date": "ds", "picu_utilization_pct": "y"}
)

# Make sure we have enough history
if len(df_ts) < 50:
    st.error(
        f"Not enough history for {selected_state} to train a robust time-series model. "
        "Pick a different state with more data."
    )
    st.stop()

# Simple train / test split
test_len = min(30, len(df_ts) // 5)
train_df = df_ts.iloc[:-test_len].copy()
test_df = df_ts.iloc[-test_len:].copy()

train_ts = train_df.set_index("ds")["y"]
test_ts = test_df.set_index("ds")["y"]

# ---------- FORECASTING MODEL SELECTOR ----------
model_choice = st.radio(
    "Forecasting model",
    options=["SARIMAX (default)", "Prophet (experimental)"],
    horizontal=True,
)

# For now we always fit SARIMAX so the rest of the code still works.
# Prophet option is just a UI toggle (no crash, but same forecasts).
if model_choice == "Prophet (experimental)":
    st.info(
        "Prophet option is enabled in the UI, but forecasts are currently "
        "generated with SARIMAX behind the scenes to keep the app stable."
    )

# ---------- FIT SARIMAX MODEL ----------
with st.spinner(f"Training SARIMAX model for {selected_state}..."):
    model = sm.tsa.SARIMAX(
        train_ts,
        order=(1, 0, 1),
        seasonal_order=(0, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)

# ---------- FORECAST ----------
future_index = pd.date_range(
    start=df_ts["ds"].iloc[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="D",
)
future_forecast = results.forecast(steps=forecast_days)

forecast_df = pd.DataFrame(
    {"date": future_index, "forecast_utilization_pct": future_forecast.values}
).set_index("date")

# Scenario-adjusted forecast
baseline = forecast_df["forecast_utilization_pct"]
demand_factor = 1 + demand_shock_pct / 100.0
capacity_factor = 1 + capacity_change_pct / 100.0 if (1 + capacity_change_pct / 100.0) != 0 else 1.0
scenario_series = baseline * demand_factor / capacity_factor
scenario_df = forecast_df.copy()
scenario_df["scenario_utilization_pct"] = scenario_series.clip(lower=0)
# ======== EVALUATION ========
# Forecast the held-out test horizon

# Number of test steps already computed earlier
# test_len should already exist above

pred_test = results.forecast(steps=test_len)

# Convert to arrays for sklearn
y_true = test_ts.values
y_pred = np.array(pred_test)

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
# ---------- KPI SUMMARY (LIVE COUNTS) ----------
latest_row = state_df.iloc[-1]
latest_util = latest_row["picu_utilization_pct"]
latest_occ = int(latest_row["staffed_pediatric_icu_bed_occupancy"])
latest_cap = int(latest_row["total_staffed_pediatric_icu_beds"])
available_now = latest_cap - latest_occ

last7 = state_df["picu_utilization_pct"].iloc[-7:].mean()
if len(state_df) >= 14:
    prev7 = state_df["picu_utilization_pct"].iloc[-14:-7].mean()
    delta7 = last7 - prev7
else:
    delta7 = np.nan
# Make scenario shocks available for KPI chips
demand_shock = float(demand_shock_pct)
capacity_shock = float(capacity_change_pct)
# ---------- RISK LEVEL BADGE ----------
if latest_util < 70:
    risk_label = "Comfortable"
    risk_emoji = "🟢"
    risk_class = "badge-risk-low"
elif latest_util < 85:
    risk_label = "Watchful"
    risk_emoji = "🟡"
    risk_class = "badge-risk-med"
elif latest_util < 95:
    risk_label = "High Risk"
    risk_emoji = "🟠"
    risk_class = "badge-risk-high"
else:
    risk_label = "Critical"
    risk_emoji = "🔴"
    risk_class = "badge-risk-high"
# ----- SPARKLINE DATA (last 14 days) -----
spark_df = state_df.tail(14)[["date", "picu_utilization_pct"]].copy()
spark_df = spark_df.set_index("date")

spark_fig = go.Figure()
spark_fig.add_trace(
    go.Scatter(
        x=spark_df.index,
        y=spark_df["picu_utilization_pct"],
        mode="lines",
        line=dict(width=2),
        hovertemplate="%{x|%b %d}: %{y:.1f}%<extra></extra>",
    )
)

spark_fig.update_layout(
    height=60,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        rangemode="tozero",
    ),
)
# ---------- ICU RISK GAUGE FIGURE ----------
gauge_fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=latest_util,
        number={"suffix": "%"},
        title={"text": "PICU Utilization Risk", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#111827"},
            "bgcolor": "white",
            "borderwidth": 0,
            "bordercolor": "white",
            "steps": [
                {"range": [0, 70], "color": "#dcfce7"},   # green
                {"range": [70, 85], "color": "#fef9c3"},  # yellow
                {"range": [85, 95], "color": "#fed7aa"},  # orange
                {"range": [95, 100], "color": "#fecaca"}  # red
            ],
            "threshold": {
                "line": {"color": "#111827", "width": 3},
                "thickness": 0.75,
                "value": latest_util,
            },
        },
    )
)

gauge_fig.update_layout(
    margin=dict(l=10, r=10, t=30, b=0),
    height=220,
)
# ---------- TOP KPI ROW: APPLE-STYLE CARDS + GAUGE ----------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns([1.1, 1.1, 1.2, 1.4])

with kpi_col1:
    st.markdown(
        f"""
        <div class="kpi-chip">
            <div class="kpi-label">State</div>
            <div class="kpi-value">{selected_state}</div>
            <div class="kpi-sub">PICU utilization snapshot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="kpi-chip">
          <div class="kpi-label">Latest utilization</div>
          <div class="kpi-value">{latest_util:.1f}%</div>
          <div class="kpi-sub">7-day avg: {last7:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        spark_fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

with kpi_col3:
    change_text = "No prior week" if np.isnan(delta7) else f"{delta7:+.1f} pts vs prev 7 days"
    st.markdown(
        f"""
        <div class="kpi-chip">
            <div class="kpi-label">Beds available now</div>
            <div class="kpi-value">{available_now}</div>
            <div class="kpi-sub">out of {latest_cap} staffed • {change_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    # Risk badge + scenario chips + small gauge
    st.markdown(
        f"""
        <div class="kpi-chip">
            <div class="kpi-label">Current risk & scenario</div>
            <div class="kpi-value">{risk_emoji} {risk_label}</div>
            <div class="kpi-sub">
                <span class="scenario-chip scenario-chip-demand">
                    Δ demand: {demand_shock:+.0f}%
                </span>
                <span class="scenario-chip scenario-chip-capacity">
                    Δ capacity: {capacity_change_pct:+.0f}%
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        gauge_fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )
# ---------- MAIN CONTENT SECTION TITLE ----------
st.markdown("---")
st.markdown(
    """
    <div class="section-title">
        <span>📊 Utilization trends & forecasting</span>
    </div>
    """,
    unsafe_allow_html=True,
)
# ---------- LATEST SNAPSHOT FOR ALL STATES (FOR MAP) ----------
latest_by_state = (
    full_df.sort_values("date")
    .groupby("state")
    .tail(1)
    .copy()
)
latest_by_state["available_beds"] = (
    latest_by_state["total_staffed_pediatric_icu_beds"]
    - latest_by_state["staffed_pediatric_icu_bed_occupancy"]
)
latest_by_state["availability_pct"] = (
    latest_by_state["available_beds"]
    / latest_by_state["total_staffed_pediatric_icu_beds"]
    * 100.0
)

# ---------- TABS LAYOUT ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Overview & Maps", "🔮 Forecast & Scenarios", "📊 Data Explorer", "🤖 CareBot Assistant"]
)

# ----- TAB 1: OVERVIEW & MAPS -----
with tab1:
    st.subheader(f"Historical Pediatric ICU Utilization – {selected_state}")

    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    ax_hist.plot(state_df["date"], state_df["picu_utilization_pct"])
    ax_hist.set_xlabel("Date")
    ax_hist.set_ylabel("PICU Utilization (%)")
    ax_hist.grid(True)
    st.pyplot(fig_hist)

    st.markdown(
        """
        **Interpretation tip (not clinical guidance):**  
        In many systems, sustained utilization above ~85–90% may indicate operational
        strain on PICU capacity and reduced flexibility to absorb new surges.
        Always consult hospital operations and clinical leadership for decisions.
        """
    )

    st.markdown("### 🗺 US Map – Pediatric ICU Availability (%)")

    fig_map = px.choropleth(
        latest_by_state,
        locations="state",
        locationmode="USA-states",
        color="availability_pct",
        color_continuous_scale="RdYlGn",
        scope="usa",
        labels={"availability_pct": "Available (%)"},
        hover_name="state",
        hover_data={
            "available_beds": True,
            "total_staffed_pediatric_icu_beds": True,
            "picu_utilization_pct": ":.2f",
        },
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # Hospital-level map for selected state
    st.markdown(f"### 🏥 Hospital-level View – {selected_state}")
    if facility_df is None:
        st.info("Add the facility-level file to enable hospital-level mapping.")
    else:
        fac_state = facility_df[facility_df["state"] == selected_state].copy()
        if fac_state.empty or fac_state["lat"].isna().all():
            st.info(
                "No facility-level coordinates available for this state or dataset. "
                "Check column names (lat/lon) in the facility CSV."
            )
        else:
            # Use most recent date per facility (if date exists)
            if "date" in fac_state.columns:
                fac_state = (
                    fac_state.sort_values("date")
                    .groupby("hospital_name" if "hospital_name" in fac_state.columns else "hospital_pk")
                    .tail(1)
                )

            fig_fac = px.scatter_geo(
                fac_state,
                lat="lat",
                lon="lon",
                scope="usa",
                color="facility_capacity_proxy",
                size="facility_capacity_proxy",
                hover_name="hospital_name" if "hospital_name" in fac_state.columns else None,
                labels={"facility_capacity_proxy": "Capacity proxy"},
            )
            fig_fac.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_fac, use_container_width=True)

# ----- TAB 2: FORECAST & SCENARIOS -----
with tab2:
    st.subheader(
        f"Baseline vs Scenario Forecast – Next {forecast_days} Days ({selected_state})"
    )

    fig_fcast, ax_fcast = plt.subplots(figsize=(12, 5))
    ax_fcast.plot(train_df["ds"], train_df["y"], label="Train", linewidth=1.5)
    ax_fcast.plot(test_df["ds"], test_df["y"], label="Test", linewidth=1.5)
    ax_fcast.plot(
        forecast_df.index,
        forecast_df["forecast_utilization_pct"],
        linestyle="--",
        linewidth=2,
        label="Baseline Forecast",
    )
    ax_fcast.plot(
        scenario_df.index,
        scenario_df["scenario_utilization_pct"],
        linestyle=":",
        linewidth=2,
        label="Scenario Forecast",
    )
    ax_fcast.legend()
    ax_fcast.grid(True)
    ax_fcast.set_xlabel("Date")
    ax_fcast.set_ylabel("PICU Utilization (%)")

    st.pyplot(fig_fcast)

    st.markdown("#### Model Performance (Last 30 Days Test)")
    st.write(f"- **MAE:** `{mae:.3f}`")
    st.write(f"- **RMSE:** `{rmse:.3f}`")

    st.markdown("#### Scenario Explanation")
    st.write(
        f"- Demand shock: **{demand_shock_pct:+d}%** change in expected admissions\n"
        f"- Capacity change: **{capacity_change_pct:+d}%** change in staffed PICU beds\n"
        "Scenario utilization is approximated as: baseline × (1 + demand%) / (1 + capacity%). "
        "This is a simplification for planning purposes only."
    )

    st.markdown("#### Baseline Forecast Table")
    st.dataframe(
        forecast_df.style.format({"forecast_utilization_pct": "{:.2f}"})
    )

    st.markdown("#### Scenario Forecast Table")
    st.dataframe(
        scenario_df[["scenario_utilization_pct"]].style.format(
            {"scenario_utilization_pct": "{:.2f}"}
        )
    )

    csv_baseline = (
        forecast_df.reset_index()
        .rename(columns={"date": "forecast_date"})
        .to_csv(index=False)
        .encode("utf-8")
    )
    csv_scenario = (
        scenario_df.reset_index()
        .rename(columns={"date": "forecast_date"})
        .to_csv(index=False)
        .encode("utf-8")
    )

    c1, c2 = st.columns(2)
    c1.download_button(
        label="⬇️ Download baseline forecast CSV",
        data=csv_baseline,
        file_name=f"carecast_baseline_{selected_state}.csv",
        mime="text/csv",
    )
    c2.download_button(
        label="⬇️ Download scenario forecast CSV",
        data=csv_scenario,
        file_name=f"carecast_scenario_{selected_state}.csv",
        mime="text/csv",
    )
    # --- Shared values for other tabs (e.g., CareBot) ---

# latest observed utilization from the time series for the selected state
latest_utilization = float(
    forecast_df["forecast_utilization_pct"].iloc[-1]
)

# scenario utilization at the end of the forecast horizon
scenario_utilization = float(
    scenario_df["scenario_utilization_pct"].iloc[-1]
)

# make sure these are simple floats for CareBot
demand_shock = float(demand_shock_pct)
capacity_shock = float(capacity_change_pct)

# ----- TAB 3: DATA EXPLORER -----
with tab3:
    st.subheader(f"Raw Data Explorer – {selected_state}")

    min_date = state_df["date"].min().date()
    max_date = state_df["date"].max().date()

    date_range = st.date_input(
        "Select date range",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    mask = (state_df["date"] >= pd.to_datetime(start_date)) & (
        state_df["date"] <= pd.to_datetime(end_date)
    )
    filtered = state_df.loc[mask].copy()

    st.write(
        f"Showing **{len(filtered)}** records from "
        f"`{start_date}` to `{end_date}` for state `{selected_state}`."
    )

    st.dataframe(
        filtered[
            [
                "date",
                "staffed_pediatric_icu_bed_occupancy",
                "total_staffed_pediatric_icu_beds",
                "picu_utilization_pct",
            ]
        ].rename(
            columns={
                "date": "Date",
                "staffed_pediatric_icu_bed_occupancy": "PICU Occupancy",
                "total_staffed_pediatric_icu_beds": "Total PICU Beds",
                "picu_utilization_pct": "Utilization (%)",
            }
        ).style.format({"Utilization (%)": "{:.2f}"})
    )


# ----- TAB 4: CAREBOT ASSISTANT -----
with tab4:
    st.subheader(f"🤖 CareBot – Capacity Planning Assistant ({selected_state})")

    st.markdown(
        """
This AI assistant helps interpret utilization and forecast trends from a **systems and operations** perspective  
(staffing, surge planning, regional coordination, risk awareness).  
It cannot give clinical advice, diagnosis, or patient-specific guidance.
        """
    )

    # ----- Default suggested prompts -----
    default_qs = [
        "What does the current utilization level imply for capacity planning?",
        "What preventive operational steps can be considered before a forecasted surge?",
        "How might staffing plans be adjusted under this scenario?",
        "How could this state coordinate with neighboring regions if beds run low?",
        "Explain the difference between baseline and scenario forecast in simple terms.",
    ]

    cols = st.columns(len(default_qs))
    for i, q in enumerate(default_qs):
        if cols[i].button(q):
            st.session_state["carebot_last_question"] = q

    st.markdown("---")

    # Use clicked button as default question if present
    preset_q = st.session_state.pop("carebot_last_question", None)
    user_question = st.text_input(
        "Ask about capacity, planning, or scenarios:",
        value=preset_q if preset_q else ""
    )
    if user_question:
     answer = get_carebot_response(
        user_question=user_question,
        state=selected_state,
        latest_util=latest_utilization,
        scenario_util=scenario_utilization,
        horizon_days=forecast_days,
        demand_shock=demand_shock,
        capacity_shock=capacity_shock,
    )
     render_carebot_answer(answer)
   

st.markdown("### 📌 Quick operational checklists for common questions")

with st.expander("What does the current utilization level imply for capacity planning?"):
    st.markdown(""" 
- Compare current utilization to internal alert thresholds (e.g., 70%, 85%, 95%).
- Check trends over the last 7–14 days (rising, stable, or falling?).
- Review available staffed beds versus surge capacity options.
- Identify constraints: staffing, physical beds, equipment, or step-down capacity.
- Confirm escalation pathways if utilization stays high.
""")

with st.expander("What preventive operational steps can be considered before a forecasted surge?"):
    st.markdown(""" 
- Pre-activate surge plans and confirm on-call staffing rosters.
- Review elective procedure schedules that could be flexed or delayed.
- Verify readiness of step-down units or alternative care areas.
- Coordinate with nearby hospitals on transfer criteria and capacity.
- Strengthen communication with clinical leadership and incident command.
""")

with st.expander("How might staffing plans be adjusted under this scenario?"):
    st.markdown(""" 
- Match staffing grids to forecasted census instead of just current census.
- Cross-train or redeploy staff from lower-acuity areas if safe and appropriate.
- Plan for fatigue management: breaks, shift length, and rotation fairness.
- Identify critical roles needing redundancy (charge nurse, RT, intensivist coverage).
- Align schedules with expected peak hours/days from the forecast.
""")

with st.expander("How could this state coordinate with neighboring regions if beds run low?"):
    st.markdown(""" 
- Maintain an updated view of regional PICU capacity where possible.
- Establish or renew transfer agreements with nearby facilities.
- Define clinical and operational criteria for inter-facility transfers.
- Use shared dashboards or agreed communication channels during surges.
- Align with state or regional emergency management structures where applicable.
""")

with st.expander("Explain the difference between baseline and scenario forecast in simple terms."):
    st.markdown(""" 
- **Baseline forecast:** Expected utilization if conditions stay as they are now.
- **Scenario forecast:** Utilization if demand or capacity changes.
- Demand shocks change how much PICU demand arrives (e.g., surge in admissions).
- Capacity shocks change how much strain the same demand creates on available beds.
- Use scenarios to stress-test plans, not as exact predictions of the future.
""")






        
