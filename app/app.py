import os
import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pymongo import MongoClient
from dotenv import load_dotenv
import warnings
import io

warnings.filterwarnings("ignore")

# Load Environment Variables
load_dotenv()

# CONFIGURATION
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")

MODEL_NAME = "AQI_MultiOutput_Predictor"
ALIAS = "champion"

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "aqi_predictor")
RAW_COLLECTION = "raw_data"
FEATURE_COLLECTION = "feature_store"
CITY_NAME = os.getenv("CITY_NAME", "Lahore")
LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "Asia/Karachi")

# PAGE SETTINGS
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    layout="wide",
    page_icon="🧭",
    initial_sidebar_state="expanded",
)

# MODERN CSS STYLING
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=DM+Sans:wght@400;500;700&display=swap');

    :root {
        --bg-0: #f8f4ec;
        --bg-1: #ece5d8;
        --ink-1: #0f2a43;
        --ink-2: #305778;
        --mint: #00a878;
        --orange: #ff7f11;
        --panel: rgba(255, 255, 255, 0.84);
        --line: rgba(15, 42, 67, 0.16);
        --shadow-soft: 0 12px 22px rgba(14, 39, 62, 0.1);
    }

    .stApp, .main {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink-1);
        background:
            radial-gradient(circle at 8% 8%, rgba(255, 127, 17, 0.18), transparent 20%),
            radial-gradient(circle at 93% 12%, rgba(0, 168, 120, 0.18), transparent 26%),
            linear-gradient(140deg, var(--bg-0) 0%, var(--bg-1) 100%);
    }

    .block-container {
        padding-top: 3.25rem;
    }

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 90% 4%, rgba(255, 127, 17, 0.22), transparent 35%),
            linear-gradient(165deg, #08263b 0%, #123954 45%, #205375 100%);
        border-right: 2px solid rgba(255, 255, 255, 0.18);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: #eff7ff !important;
    }

    /* Keep Material Symbols font untouched so icon ligatures don't render as words. */
    section[data-testid="stSidebar"] .material-symbols-rounded,
    section[data-testid="stSidebar"] [class*="material-symbol"] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-smoothing: antialiased !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
    }

    .main-header {
        background:
            linear-gradient(120deg, #0f2a43 0%, #183f5f 55%, #245f74 100%),
            repeating-linear-gradient(
                120deg,
                rgba(255, 255, 255, 0.12) 0,
                rgba(255, 255, 255, 0.12) 8px,
                rgba(255, 255, 255, 0.02) 8px,
                rgba(255, 255, 255, 0.02) 22px
            );
        border: 2px solid rgba(255, 255, 255, 0.34);
        padding: 2.4rem 2rem 2rem 2.2rem;
        border-radius: 12px;
        color: #fdfdfd;
        text-align: left;
        margin-bottom: 2rem;
        margin-top: 0.5rem;
        box-shadow: 0 16px 28px rgba(15, 42, 67, 0.24);
        position: relative;
        overflow: hidden;
        animation: reveal-header 640ms cubic-bezier(.25,.8,.25,1);
    }

    .main-header::after {
        content: "";
        position: absolute;
        width: 150px;
        height: 150px;
        right: -35px;
        top: -30px;
        border-radius: 50%;
        border: 2px dashed rgba(255, 255, 255, 0.45);
        opacity: 0.8;
    }

    .main-header h1 {
        margin: 0;
        font-size: clamp(1.9rem, 4vw, 2.65rem);
        letter-spacing: 0.015em;
        font-family: 'Archivo Black', sans-serif;
        font-weight: 400;
        line-height: 1.1;
    }

    .main-header p {
        margin: 0.65rem 0 0 0;
        font-size: 0.98rem;
        opacity: 0.96;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        color: #4c687f !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.12rem !important;
        font-family: 'Archivo Black', sans-serif !important;
        font-weight: 700 !important;
        color: var(--ink-1) !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.79rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="metric-container"] {
        background: var(--panel);
        border: 1.5px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow-soft);
        padding: 1.1rem 0.95rem;
        min-height: 118px;
        position: relative;
        overflow: hidden;
        clip-path: polygon(0 0, 94% 0, 100% 14%, 100% 100%, 0 100%);
        transition: transform 220ms ease, box-shadow 220ms ease;
    }

    div[data-testid="metric-container"]::after {
        content: "";
        position: absolute;
        inset: auto 0 0 0;
        height: 4px;
        background: linear-gradient(90deg, var(--orange), #ffd166, var(--mint));
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) rotate(-0.4deg);
        box-shadow: 0 16px 28px rgba(15, 42, 67, 0.16);
    }

    .stButton > button,
    .stDownloadButton > button {
        border: 1.5px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        padding: 0.72rem 1.5rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        letter-spacing: 0.055em;
        text-transform: uppercase;
        color: #ffffff;
        background: linear-gradient(110deg, #ff7f11 0%, #ff5a3d 45%, #00a878 100%);
        box-shadow: 0 10px 20px rgba(15, 42, 67, 0.2);
        transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        filter: contrast(1.04) saturate(1.06);
        box-shadow: 0 14px 26px rgba(15, 42, 67, 0.24);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        background: rgba(255, 255, 255, 0.74);
        border: 1.5px solid var(--line);
        border-radius: 10px;
        padding: 0.35rem;
        box-shadow: var(--shadow-soft);
    }

    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 8px;
        padding: 0 16px;
        color: var(--ink-2);
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(15, 42, 67, 0.06);
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        letter-spacing: 0.02em;
        transition: all 170ms ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        border-color: rgba(0, 168, 120, 0.44);
        color: var(--ink-1);
        transform: translateY(-1px);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(100deg, rgba(15, 42, 67, 0.95), rgba(0, 168, 120, 0.92));
        color: #fefefe !important;
        border-color: rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 16px rgba(15, 42, 67, 0.25);
    }

    .alert-card {
        background: linear-gradient(115deg, rgba(255, 255, 255, 0.94), rgba(247, 238, 221, 0.92));
        border: 1.5px solid rgba(15, 42, 67, 0.12);
        padding: 1.15rem 1.25rem;
        border-radius: 8px;
        margin: 0.9rem 0;
        box-shadow: 0 10px 18px rgba(15, 42, 67, 0.08);
    }

    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 8px;
        border-left-width: 5px;
        border-top: 1px solid rgba(15, 42, 67, 0.08);
        border-right: 1px solid rgba(15, 42, 67, 0.08);
        border-bottom: 1px solid rgba(15, 42, 67, 0.08);
    }

    @media (max-width: 900px) {
        .main-header {
            padding: 1.45rem 1rem;
            border-radius: 10px;
        }

        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            white-space: nowrap;
            flex-wrap: nowrap;
        }
    }

    @keyframes reveal-header {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
""",
    unsafe_allow_html=True,
)


# CACHED MODEL LOADING WITH AUTO-UPDATE
@st.cache_resource(show_spinner=False)
def load_champion_model():
    """Loads the model from MLflow Registry with error handling."""
    try:
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            raise ValueError("MLflow tracking URI not configured")

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        model_uri = f"models:/{MODEL_NAME}@{ALIAS}"

        model = mlflow.pyfunc.load_model(model_uri)
        client = mlflow.tracking.MlflowClient()
        model_ver = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

        return model, model_ver.version, None
    except Exception as e:
        return None, None, str(e)


def check_for_model_updates(current_version):
    """
    Check if a new champion model version is available in the registry.
    Returns (has_update, new_version, error)
    """
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = mlflow.tracking.MlflowClient()

        # Get the latest champion model version
        latest_model = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        latest_version = latest_model.version

        # Check if there's a new version
        if latest_version != current_version:
            return True, latest_version, None

        return False, current_version, None

    except Exception as e:
        return False, current_version, str(e)


# UTILITY FUNCTIONS
def calculate_aqi(pm25):
    """Calculate AQI from PM2.5 concentration using EPA formula."""
    if pm25 < 0:
        return 0
    if pm25 <= 12.0:
        return round(((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0)
    elif pm25 <= 35.4:
        return round(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4:
        return round(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4:
        return round(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4:
        return round(((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201)
    else:
        return 500


def normalize_forecast_pm25(pred_output, horizon=3, fallback_pm25=0.0):
    """
    Normalize model output to a fixed forecast horizon.

    Handles older/single-output models and non-finite values by padding
    with a sensible fallback so UI metrics never break.
    """
    pred_arr = np.asarray(pred_output, dtype=float).reshape(-1)
    pred_arr = np.maximum(pred_arr, 0)
    pred_arr = pred_arr[np.isfinite(pred_arr)]

    if pred_arr.size == 0:
        return np.full(horizon, max(float(fallback_pm25), 0.0), dtype=float)

    if pred_arr.size < horizon:
        pad_val = float(pred_arr[-1])
        pred_arr = np.concatenate(
            [pred_arr, np.full(horizon - pred_arr.size, pad_val, dtype=float)]
        )
    elif pred_arr.size > horizon:
        pred_arr = pred_arr[:horizon]

    return pred_arr


def get_aqi_info(aqi_val):
    """Get AQI category, color, and health recommendation."""
    if aqi_val <= 50:
        return (
            "Good",
            "#00e400",
            "☑️ Air quality is satisfactory. Ideal for all outdoor activities!",
        )
    elif aqi_val <= 100:
        return (
            "Moderate",
            "#ffff00",
            "ℹ️ Air quality is acceptable. Sensitive groups should limit prolonged outdoor exertion.",
        )
    elif aqi_val <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            "#ff7e00",
            "⚕️ Members of sensitive groups may experience health effects. Wear a mask if needed.",
        )
    elif aqi_val <= 200:
        return (
            "Unhealthy",
            "#ff0000",
            "🚫 Everyone may experience health effects. Reduce prolonged outdoor activities.",
        )
    elif aqi_val <= 300:
        return (
            "Very Unhealthy",
            "#8f3f97",
            "🔴 Health alert: Everyone should avoid all outdoor physical activity.",
        )
    else:
        return (
            "Hazardous",
            "#7e0023",
            "☣️ Emergency conditions. Stay indoors with air purification systems.",
        )


def create_aqi_chart_plotly(plot_dates, aqi_values, types):
    """An interactive AQI visualization chart using Plotly."""

    # Convert datetime objects to ensure compatibility
    plot_dates = [
        pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d for d in plot_dates
    ]

    # Create figure with better styling
    fig = go.Figure()

    # Add AQI category background bands with improved styling
    aqi_bands = [
        (0, 50, "#2ec4b6", "Good"),
        (51, 100, "#ffbf69", "Moderate"),
        (101, 150, "#ff9f1c", "Unhealthy (Sensitive)"),
        (151, 200, "#e71d36", "Unhealthy"),
        (201, 300, "#8338ec", "Very Unhealthy"),
        (301, 500, "#5a189a", "Hazardous"),
    ]

    for low, high, color, label in aqi_bands:
        fig.add_hrect(
            y0=low,
            y1=high,
            fillcolor=color,
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text=label,
            annotation_position="right",
            annotation=dict(
                font_size=10,
                font_color=color,
                font_family="DM Sans, sans-serif",
            ),
        )

    # Add vertical line separating observed and predicted
    separator_idx = types.index("Predicted") if "Predicted" in types else len(types)
    if separator_idx < len(plot_dates):
        # Use shapes instead of add_vline to avoid datetime arithmetic issues
        fig.add_shape(
            type="line",
            x0=plot_dates[separator_idx],
            x1=plot_dates[separator_idx],
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#0f2a43", width=2.5, dash="dashdot"),
            opacity=0.72,
        )
        # Add annotation separately
        fig.add_annotation(
            x=plot_dates[separator_idx],
            y=1,
            yref="paper",
            text="◆ Forecast Starts",
            showarrow=False,
            yshift=15,
            font=dict(
                size=11,
                color="#0f2a43",
                family="DM Sans, sans-serif",
                weight="bold",
            ),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ff7f11",
            borderwidth=1.5,
            borderpad=4,
        )

    # Split data into observed and predicted
    observed_dates = [d for d, t in zip(plot_dates, types) if t == "Observed"]
    observed_aqi = [a for a, t in zip(aqi_values, types) if t == "Observed"]
    predicted_dates = [d for d, t in zip(plot_dates, types) if t == "Predicted"]
    predicted_aqi = [a for a, t in zip(aqi_values, types) if t == "Predicted"]

    # Add main trend line (observed) with gradient effect
    if observed_dates:
        fig.add_trace(
            go.Scatter(
                x=observed_dates,
                y=observed_aqi,
                mode="lines+markers",
                name="Observed",
                line=dict(color="#00a878", width=3.8, shape="hv"),
                marker=dict(
                    size=13,
                    color=[get_aqi_info(a)[1] for a in observed_aqi],
                    line=dict(color="#ffffff", width=2.2),
                    symbol="square",
                ),
                fill="tozeroy",
                fillcolor="rgba(0, 168, 120, 0.18)",
                hovertemplate="<b>🗓️ %{x|%b %d, %Y}</b><br>📏 AQI: <b>%{y}</b><br>🏷️ %{text}<extra></extra>",
                text=[get_aqi_info(a)[0] for a in observed_aqi],
            )
        )

    # Add predicted trend line with different styling
    if predicted_dates:
        # Connect last observed to first predicted
        connect_dates = (
            [observed_dates[-1], predicted_dates[0]]
            if observed_dates
            else predicted_dates
        )
        connect_aqi = (
            [observed_aqi[-1], predicted_aqi[0]] if observed_aqi else predicted_aqi
        )

        fig.add_trace(
            go.Scatter(
                x=connect_dates + predicted_dates[1:],
                y=connect_aqi + predicted_aqi[1:],
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#ff7f11", width=3.6, dash="dash", shape="linear"),
                marker=dict(
                    size=15,
                    symbol="triangle-up",
                    color=[get_aqi_info(a)[1] for a in connect_aqi + predicted_aqi[1:]],
                    line=dict(color="#ffffff", width=2.2),
                ),
                fill="tozeroy",
                fillcolor="rgba(255, 127, 17, 0.14)",
                hovertemplate="<b>🗓️ %{x|%b %d, %Y}</b><br>🔭 Forecast AQI: <b>%{y}</b><br>🏷️ %{text}<extra></extra>",
                text=[get_aqi_info(a)[0] for a in connect_aqi + predicted_aqi[1:]],
            )
        )

    # Update layout with modern styling
    fig.update_layout(
        title={
            "text": "AQI Signal Timeline | Historical + Forecast",
            "x": 0.02,
            "xanchor": "left",
            "font": {
                "size": 22,
                "color": "#0f2a43",
                "family": "Archivo Black, sans-serif",
                "weight": "bold",
            },
        },
        xaxis_title="Date",
        yaxis_title="AQI Index",
        hovermode="x unified",
        plot_bgcolor="#fffcf6",
        paper_bgcolor="rgba(0,0,0,0)",
        height=550,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#0f2a43",
            borderwidth=1.3,
            font=dict(size=11, family="DM Sans, sans-serif"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(15, 42, 67, 0.1)",
            tickformat="%b %d\n%a",
            tickfont=dict(size=10, color="#305778", family="DM Sans, sans-serif"),
            linecolor="#0f2a43",
            linewidth=1.3,
            showline=True,
            mirror=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(15, 42, 67, 0.12)",
            range=[0, max(aqi_values) + 50],
            tickfont=dict(size=10, color="#305778", family="DM Sans, sans-serif"),
            linecolor="#0f2a43",
            linewidth=1.3,
            showline=True,
            mirror=True,
            zeroline=True,
            zerolinecolor="rgba(15, 42, 67, 0.18)",
            zerolinewidth=1.2,
        ),
        font=dict(family="DM Sans, sans-serif", size=12, color="#0f2a43"),
    )

    return fig


def create_historical_overview_chart(hist_df):
    """Create multi-parameter historical overview chart."""

    # Map column names from database to display names
    column_mapping = {
        "temperature": "temp",
        "humidity": "rh",
        "wind_speed": "ws",
        "pressure": "pres",
    }

    # Rename columns if they exist in old format
    for old_name, new_name in column_mapping.items():
        if old_name in hist_df.columns and new_name not in hist_df.columns:
            hist_df[new_name] = hist_df[old_name]

    # Create subplots with better styling - 2x2 grid for 4 parameters
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "PM2.5 Concentration",
            "PM10 Concentration",
            "Temperature",
            "Humidity",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Define parameters with modern, vibrant colors and their RGBA equivalents
    params = [
        ("pm2_5", "PM2.5 (μg/m³)", "#ef476f", "rgba(239, 71, 111, 0.2)", 1, 1),
        ("pm10", "PM10 (μg/m³)", "#3a86ff", "rgba(58, 134, 255, 0.2)", 1, 2),
        ("temp", "Temperature (°C)", "#ff7f11", "rgba(255, 127, 17, 0.18)", 2, 1),
        ("rh", "Humidity (%)", "#00a878", "rgba(0, 168, 120, 0.2)", 2, 2),
    ]

    for param, label, color, fill_color, row, col in params:
        if param in hist_df.columns:
            # Create gradient effect with area fill
            fig.add_trace(
                go.Scatter(
                    x=hist_df["datetime"],
                    y=hist_df[param],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=3.2, shape="hv"),
                    fill="tozeroy",
                    fillcolor=fill_color,
                    showlegend=False,
                    hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>"
                    + f"{label}: %{{y:.2f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            # Add subtle trend line
            if len(hist_df) > 10:
                # Calculate simple moving average for trend
                window = min(24, len(hist_df) // 4)
                trend = hist_df[param].rolling(window=window, center=True).mean()

                fig.add_trace(
                    go.Scatter(
                        x=hist_df["datetime"],
                        y=trend,
                        mode="lines",
                        line=dict(color="#0f2a43", width=1.5, dash="dash"),
                        showlegend=False,
                        hovertemplate="<b>Trend</b><br>%{y:.2f}<extra></extra>",
                        opacity=0.45,
                    ),
                    row=row,
                    col=col,
                )

    # layout
    fig.update_layout(
        height=700,
        showlegend=False,
        title={
            "text": "Environmental Pulse | Last 7 Days",
            "x": 0.02,
            "xanchor": "left",
            "font": {
                "size": 22,
                "color": "#0f2a43",
                "family": "Archivo Black, sans-serif",
                "weight": "bold",
            },
        },
        plot_bgcolor="#fffcf6",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        font=dict(family="DM Sans, sans-serif", size=12, color="#0f2a43"),
    )

    # subplot titles
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(
            size=13,
            color="#0f2a43",
            family="DM Sans, sans-serif",
            weight="bold",
        )

    # axes with better styling
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(15, 42, 67, 0.1)",
        tickfont=dict(size=10, color="#305778", family="DM Sans, sans-serif"),
        linecolor="#0f2a43",
        linewidth=1.2,
        showline=True,
        mirror=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15, 42, 67, 0.12)",
        tickfont=dict(size=10, color="#305778", family="DM Sans, sans-serif"),
        linecolor="#0f2a43",
        linewidth=1.2,
        showline=True,
        mirror=True,
        zeroline=True,
        zerolinecolor="rgba(15, 42, 67, 0.2)",
        zerolinewidth=1.1,
    )

    return fig


def create_download_dataframe(plot_dates, aqi_values, types, forecast_pm25=None):
    """Create comprehensive dataframe for download."""
    # Convert to pandas Timestamp if not already
    plot_dates_ts = [
        pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d for d in plot_dates
    ]

    df = pd.DataFrame(
        {
            "Date": [d.strftime("%Y-%m-%d") for d in plot_dates_ts],
            "Day": [d.strftime("%A") for d in plot_dates_ts],
            "Time": [d.strftime("%H:%M:%S") for d in plot_dates_ts],
            "AQI": aqi_values,
            "Category": [get_aqi_info(a)[0] for a in aqi_values],
            "Type": types,
            "Health_Recommendation": [get_aqi_info(a)[2] for a in aqi_values],
        }
    )

    # Add PM2.5 values for predicted days if available
    if forecast_pm25 is not None:
        pm25_values = [None] * (len(plot_dates) - len(forecast_pm25)) + list(
            forecast_pm25
        )
        df["PM2.5_Forecast"] = pm25_values

    return df


# MAIN APP
def main():
    # Initialize session state for update tracking
    if "update_checked" not in st.session_state:
        st.session_state.update_checked = False
        st.session_state.update_available = False
        st.session_state.new_version = None

    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🧭 Air Quality Intelligence Dashboard</h1>
            <p>Real-time AQI Monitoring & AI-Powered 3-Day Forecast</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ System Status")

        # Load model
        with st.spinner("Loading AI model..."):
            model, version, error = load_champion_model()

        if error:
            st.error(f"⛔ Model Loading Failed\n\n{error}")
            st.stop()
        else:
            st.success("🟢 Model Active")
            st.info(f"**Model:** {MODEL_NAME}")
            st.info(f"**Version:** v{version}")
            st.info(f"**Registry:** DagsHub MLflow")

            # Auto-check for model updates on every app load
            if not st.session_state.update_checked:
                has_update, new_version, update_error = check_for_model_updates(version)
                st.session_state.update_checked = True
                st.session_state.update_available = has_update
                st.session_state.new_version = new_version

            # Display update notification if available
            if st.session_state.update_available:
                st.warning(f"📢 New model available: v{st.session_state.new_version}")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("🔼 Update Now", width="stretch", type="primary"):
                        with st.spinner("Updating model..."):
                            st.cache_resource.clear()
                            st.session_state.update_available = False
                            st.session_state.update_checked = False
                            st.rerun()

                with col2:
                    if st.button("⏩ Skip", width="stretch"):
                        st.session_state.update_available = False
                        st.rerun()
            else:
                st.caption("✓ Model up to date")

        st.markdown("---")

        # Manual refresh button
        if st.button("♻️ Refresh Model", width="stretch"):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### 🧾 About")
        st.markdown(
            """
        This dashboard provides:
        - **Real-time** air quality monitoring
        - **AI-powered** 3-day forecasts
        - **Interactive** Plotly visualizations
        - **Auto-update** model detection
        - **Health recommendations** based on AQI levels
        - **Historical trend** analysis
        - **Downloadable** reports
        """
        )

        st.markdown("---")
        st.caption("Developed by Shehryar Naveed")
        st.caption("(a.k.a Data Scientist)")

    # Main content
    st.markdown(f"### 🗺️ Latest Air Quality Analysis | {CITY_NAME}")

    # Prediction button
    if st.button("✨ Generate AI Forecast", width="stretch", type="primary"):
        local_tz = ZoneInfo(LOCAL_TIMEZONE)

        with st.spinner("🗄️ Fetching data from MongoDB..."):
            try:
                # Connect to MongoDB
                client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
                db = client[DB_NAME]
                raw_col = db[RAW_COLLECTION]
                feature_col = db[FEATURE_COLLECTION]
                city_query = {"city": CITY_NAME}

                # Fetch historical data (7 days)
                history = list(
                    raw_col.find(city_query).sort("datetime", -1).limit(168)
                )  # 7 days hourly
                if not history:
                    st.error(
                        f"⛔ No historical data found in database for {CITY_NAME}. Run the hourly pipeline for this city first."
                    )
                    client.close()
                    st.stop()

                hist_df = pd.DataFrame(history).sort_values("datetime")
                hist_df["datetime"] = pd.to_datetime(hist_df["datetime"], utc=True)
                hist_df["datetime_local"] = hist_df["datetime"].dt.tz_convert(local_tz)
                present_time = hist_df["datetime_local"].iloc[-1]

                # Fetch latest features
                latest_feat_doc = list(
                    feature_col.find(city_query).sort("datetime", -1).limit(1)
                )
                if not latest_feat_doc:
                    st.error(
                        f"⛔ No feature data found in database for {CITY_NAME}. Run feature engineering for this city first."
                    )
                    client.close()
                    st.stop()

                latest_feat_doc = latest_feat_doc[0]
                client.close()

            except Exception as e:
                st.error(f"⛔ Database Error: {str(e)}")
                st.stop()

            with st.spinner("🧠 Running AI prediction model..."):
                try:
                    # Prepare features
                    feat_df = pd.DataFrame([latest_feat_doc]).drop(
                        columns=[
                            "_id",
                            "city",
                            "datetime",
                            "target_h24",
                            "target_h48",
                            "target_h72",
                        ],
                        errors="ignore",
                    )

                    # Make predictions and normalize to 3-day horizon.
                    raw_pred = model.predict(feat_df)
                    fallback_pm25 = (
                        float(hist_df["pm2_5"].tail(24).mean())
                        if "pm2_5" in hist_df.columns and not hist_df.empty
                        else 0.0
                    )
                    forecast_pm25 = normalize_forecast_pm25(
                        raw_pred, horizon=3, fallback_pm25=fallback_pm25
                    )

                except Exception as e:
                    st.error(f"⛔ Prediction Error: {str(e)}")
                    st.stop()

        # Process results
        plot_dates, aqi_values, types = [], [], []

        # Past 3 days + today (Observed) - total 4 days
        # Day -3, -2, -1, and 0 (today)
        for d in [3, 2, 1, 0]:
            t_date = (present_time - timedelta(days=d)).date()
            day_data = hist_df[hist_df["datetime_local"].dt.date == t_date]
            avg_pm = day_data["pm2_5"].mean() if not day_data.empty else 0
            aqi = calculate_aqi(avg_pm)
            # Convert to pandas Timestamp for consistency
            plot_dates.append(
                pd.Timestamp(datetime.combine(t_date, datetime.min.time()))
            )
            aqi_values.append(aqi)
            types.append("Observed")

        # Next 3 days (Predicted) - starting from tomorrow
        # Tomorrow (+1 day), Day after (+2 days), Third day (+3 days)
        for i, days_ahead in enumerate([1, 2, 3]):
            f_dt = present_time + timedelta(days=days_ahead)
            aqi = calculate_aqi(forecast_pm25[i])
            # Convert to pandas Timestamp for consistency
            plot_dates.append(pd.Timestamp(f_dt.tz_localize(None)))
            aqi_values.append(aqi)
            types.append("Predicted")

        # DISPLAY RESULTS
        st.success("🎉 Analysis Complete!")

        # Build date-indexed AQI map for reliable local-day labels.
        aqi_by_date = {}
        for dt, aqi, aqi_type in zip(plot_dates, aqi_values, types):
            d = pd.Timestamp(dt).date()
            if d not in aqi_by_date or aqi_type == "Observed":
                aqi_by_date[d] = int(aqi)

        today_date = datetime.now(local_tz).date()
        yesterday_date = today_date - timedelta(days=1)

        today_aqi = aqi_by_date.get(today_date)
        if today_aqi is None:
            today_aqi = aqi_by_date.get(yesterday_date)

        # Current status card
        if today_aqi is not None:
            cur_aqi = today_aqi
        else:
            cur_aqi = int(aqi_values[3])

        cat_name, cat_color, rec = get_aqi_info(cur_aqi)

        st.markdown(
            f"""
            <div class="alert-card" style="border-left-color: {cat_color};">
                <h3 style="color: {cat_color}; margin-top: 0;">
                    Current Air Quality: {cat_name} (AQI {cur_aqi})
                </h3>
                <p style="font-size: 1.1rem; margin-bottom: 0;">{rec}</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Key metrics
        st.markdown("### 📐 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        # Anchor metric cards to an observed date so next 3 forecast days align.
        metric_base_date = today_date
        if metric_base_date not in aqi_by_date:
            if yesterday_date in aqi_by_date:
                metric_base_date = yesterday_date
            else:
                metric_base_date = present_time.date()

        first_metric_label = "Today's AQI"
        if metric_base_date != today_date:
            first_metric_label = f"Latest AQI ({metric_base_date.strftime('%b %d')})"

        metric_targets = [
            (first_metric_label, metric_base_date),
            ("Next Day", metric_base_date + timedelta(days=1)),
            ("Day +2", metric_base_date + timedelta(days=2)),
            ("Day +3", metric_base_date + timedelta(days=3)),
        ]

        metric_values = [
            aqi_by_date.get(target_date) for _, target_date in metric_targets
        ]

        with col1:
            if metric_values[0] is not None:
                cur_aqi_cat = get_aqi_info(metric_values[0])[0]
                st.metric(
                    label=metric_targets[0][0],
                    value=f"{metric_values[0]}",
                    delta=cur_aqi_cat,
                )
            else:
                st.metric(label=metric_targets[0][0], value="N/A", delta="No data")

        with col2:
            if metric_values[1] is not None:
                metric_cat = get_aqi_info(metric_values[1])[0]
                st.metric(
                    label=metric_targets[1][0],
                    value=f"{metric_values[1]}",
                    delta=metric_cat,
                )
            else:
                st.metric(label=metric_targets[1][0], value="N/A", delta="No data")

        with col3:
            if metric_values[2] is not None:
                metric_cat = get_aqi_info(metric_values[2])[0]
                st.metric(
                    label=metric_targets[2][0],
                    value=f"{metric_values[2]}",
                    delta=metric_cat,
                )
            else:
                st.metric(label=metric_targets[2][0], value="N/A", delta="No data")

        with col4:
            if metric_values[3] is not None:
                metric_cat = get_aqi_info(metric_values[3])[0]
                st.metric(
                    label=metric_targets[3][0],
                    value=f"{metric_values[3]}",
                    delta=metric_cat,
                )
            else:
                st.metric(label=metric_targets[3][0], value="N/A", delta="No data")

        # Interactive Plotly Visualization
        st.markdown("### 📡 Interactive AQI Trend Analysis")
        fig = create_aqi_chart_plotly(plot_dates, aqi_values, types)
        st.plotly_chart(fig, width="stretch")

        # Detailed information tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🗂️ Forecast Table",
                "🩺 Health Guidance",
                "📘 Data Insights",
                "🧪 Historical Overview",
            ]
        )

        with tab1:
            st.markdown("#### Complete 7-Day AQI Report")

            # Create downloadable dataframe
            report_df = create_download_dataframe(
                plot_dates, aqi_values, types, forecast_pm25
            )

            # Display the dataframe
            st.dataframe(report_df, width="stretch", hide_index=True)

            # Download button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Convert to CSV
                csv_buffer = io.StringIO()
                report_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="💾 Download 7-Day Report (CSV)",
                    data=csv_data,
                    file_name=f"AQI_7Day_Report_{CITY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width="stretch",
                )

        with tab2:
            st.markdown("#### Personalized Health Recommendations")

            for i in range(4, 7):  # Predicted days
                d_name = plot_dates[i].strftime("%A, %B %d")
                c_name, c_color, c_rec = get_aqi_info(aqi_values[i])

                st.markdown(
                    f"""
                    <div class="alert-card" style="border-left-color: {c_color};">
                        <h4 style="color: {c_color}; margin-top: 0;">{d_name}</h4>
                        <p style="margin: 0;"><strong>Forecast:</strong> {c_name} (AQI {aqi_values[i]})</p>
                        <p style="margin: 0.5rem 0 0 0;">{c_rec}</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

        with tab3:
            st.markdown("#### Statistical Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Historical Data (Past 4 Days)**")
                hist_aqi = aqi_values[:4]
                st.write(f"• Average AQI: {np.mean(hist_aqi):.1f}")
                st.write(f"• Max AQI: {max(hist_aqi)}")
                st.write(f"• Min AQI: {min(hist_aqi)}")
                st.write(
                    f"• Trend: {'Improving ↓' if hist_aqi[-1] < hist_aqi[0] else 'Worsening ↑'}"
                )

            with col2:
                st.markdown("**Forecast (Next 3 Days)**")
                pred_aqi = aqi_values[4:]
                st.write(f"• Average AQI: {np.mean(pred_aqi):.1f}")
                st.write(f"• Max AQI: {max(pred_aqi)}")
                st.write(f"• Min AQI: {min(pred_aqi)}")
                st.write(
                    f"• Overall Outlook: {get_aqi_info(int(np.mean(pred_aqi)))[0]}"
                )

            st.markdown("---")
            st.info(
                f"🕒 Last Updated ({CITY_NAME}): {present_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Predicted PM2.5 values
            st.markdown("---")
            st.markdown("#### Predicted PM2.5 Concentrations")
            pred_df = pd.DataFrame(
                {
                    "Forecast Period": ["24 Hours", "48 Hours", "72 Hours"],
                    "PM2.5 (μg/m³)": [f"{pm:.2f}" for pm in forecast_pm25],
                    "AQI": [aqi_values[4], aqi_values[5], aqi_values[6]],
                }
            )
            st.dataframe(pred_df, width="stretch", hide_index=True)

        with tab4:
            st.markdown("#### Environmental Parameters - Historical Overview")

            # Create interactive historical chart
            hist_chart = create_historical_overview_chart(hist_df)
            st.plotly_chart(hist_chart, width="stretch")

            # Summary statistics
            st.markdown("---")
            st.markdown("#### Summary Statistics (Past 7 Days)")

            col1, col2 = st.columns(2)

            # Handle both old and new column naming conventions
            temp_col = "temp" if "temp" in hist_df.columns else "temperature"
            hum_col = "rh" if "rh" in hist_df.columns else "humidity"

            with col1:
                st.metric("Avg PM2.5", f"{hist_df['pm2_5'].mean():.2f} μg/m³")
                if temp_col in hist_df.columns:
                    st.metric("Avg Temperature", f"{hist_df[temp_col].mean():.1f}°C")

            with col2:
                st.metric("Avg PM10", f"{hist_df['pm10'].mean():.2f} μg/m³")
                if hum_col in hist_df.columns:
                    st.metric("Avg Humidity", f"{hist_df[hum_col].mean():.1f}%")

    else:
        # Initial state - show instructions
        st.info(
            "⤴️ Click the button above to generate the latest AI-powered air quality forecast"
        )

        # Show example info
        st.markdown("### 🧩 What You'll Get")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                **📐 Real-time Analysis**
                - Current AQI status
                - PM2.5 measurements
                - 24-hour trends
            """
            )

        with col2:
            st.markdown(
                """
                **🧠 AI Predictions**
                - 24h forecast
                - 48h forecast
                - 72h forecast
            """
            )

        with col3:
            st.markdown(
                """
                **🩺 Health Guidance**
                - Category-based advice
                - Activity recommendations
                - Risk assessments
            """
            )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                **🧪 Interactive Visualizations**
                - Plotly-powered charts
                - Multi-parameter analysis
                - Historical trends
            """
            )

        with col2:
            st.markdown(
                """
                **💾 Export Features**
                - Download 7-day reports
                - CSV format
                - Complete data export
            """
            )


if __name__ == "__main__":
    main()
