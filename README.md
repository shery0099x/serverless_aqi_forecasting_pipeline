# 🌫️ AQI Predictor — Pearls ML Project

> Predict the Air Quality Index (AQI) for your city **3 days ahead** using a
> fully serverless, automated ML pipeline — no servers to manage, no cloud bills.

---

## 📌 Project Overview

This project builds an end-to-end machine learning system for AQI forecasting using
a four-pipeline architecture:

| Pipeline | Trigger | What it does |
|---|---|---|
| **Feature pipeline** | Every hour (GitHub Actions) | Fetches AQI + weather data, engineers features, upserts to Feature Store |
| **Backfill pipeline** | Run once manually | Populates the Feature Store with 12–18 months of historical (features, targets) |
| **Training pipeline** | Every day (GitHub Actions) | Trains Ridge, Random Forest, XGBoost; picks the best; registers in Model Registry |
| **Dashboard** | On-demand (Streamlit) | Loads model + features, shows 3-day forecast, SHAP charts, and hazard alerts |

---

## 🏗️ Architecture

```
Weather & AQI APIs  ──►  Feature Pipeline  ──►  Feature Store (Hopsworks)
                                                        │
                         Backfill Pipeline  ────────────┤
                                                        │
                         Training Pipeline  ◄───────────┤
                                │                       │
                          Model Registry  ──────────────┤
                                                        │
                         Streamlit Dashboard  ◄─────────┘
```

---

## 📂 Project Structure

```
aqi_predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml    # Cron: every hour
│       └── training_pipeline.yml  # Cron: every day at 02:00 UTC
│
├── app/
│   └── dashboard.py               # Streamlit web dashboard
│
├── config/
│   └── settings.py                # All config loaded from .env
│
├── notebooks/
│   └── eda.py                     # Exploratory data analysis script
│
├── pipelines/
│   ├── feature_pipeline.py        # Pipeline 1 — hourly data collection
│   ├── backfill_pipeline.py       # Pipeline 2 — historical data load
│   └── training_pipeline.py      # Pipeline 3 — daily model training
│
├── tests/
│   └── test_feature_engineering.py
│
├── utils/
│   ├── aqi_helpers.py             # AQI classification + health advice
│   └── logger.py                  # Shared loguru logger
│
├── .env.example                   # Template for environment variables
├── .gitignore
├── requirements.txt
├── README.md
└── DOCUMENTATION.md               # Full technical documentation
```

---

## ⚡ Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/aqi_predictor.git
cd aqi_predictor
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Open .env and fill in your API keys (see DOCUMENTATION.md for how to get each one)
```

### 3. Run the backfill (one-time)

```bash
python -m pipelines.backfill_pipeline \
    --start 2023-01-01 \
    --end   2024-12-31 \
    --source csv \
    --csv-path data/lahore_historical_aqi.csv
```

### 4. Run the feature pipeline manually (test)

```bash
python -m pipelines.feature_pipeline
```

### 5. Train models

```bash
python -m pipelines.training_pipeline
```

### 6. Launch dashboard

```bash
streamlit run app/dashboard.py
```

---

## 🔑 Required API Keys

| Key | Where to get | Free tier |
|---|---|---|
| `AQICN_TOKEN` | https://aqicn.org/data-platform/token/ | ✅ Yes |
| `OPENWEATHER_API_KEY` | https://openweathermap.org/api | ✅ Yes (current weather) |
| `HOPSWORKS_API_KEY` | https://app.hopsworks.ai → Settings → API Keys | ✅ Free tier available |

---

## 🤖 Models Trained

Each of the three forecast horizons (24h, 48h, 72h) trains four candidates:

- **Ridge Regression** — linear baseline
- **Random Forest** — robust non-linear baseline
- **Gradient Boosting** — strong ensemble baseline
- **XGBoost** — typically the best performer

The model with the lowest RMSE on the held-out test set is registered per horizon.

---

## 📊 Features Engineered

| Category | Features |
|---|---|
| Time-based | `hour`, `day_of_week`, `month`, `is_weekend` |
| Pollutants | `pm25`, `pm10`, `o3`, `no2`, `co`, `so2` |
| Weather | `temperature`, `humidity`, `pressure`, `wind_speed`, `wind_deg`, `visibility` |
| Lag | `aqi_lag_1h`, `aqi_lag_3h`, `aqi_lag_6h`, `aqi_lag_12h`, `aqi_lag_24h` |
| Rolling | `aqi_roll_mean_3h`, `aqi_roll_mean_6h`, `aqi_roll_mean_24h` |
| Derived | `aqi_change_rate` (current − 3h-ago AQI) |

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=pipelines --cov-report=term-missing
```

---

## 🚀 CI/CD Automation

Add these secrets in **GitHub → Repo → Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `AQICN_TOKEN` | Your AQICN token |
| `OPENWEATHER_API_KEY` | Your OpenWeather key |
| `HOPSWORKS_API_KEY` | Your Hopsworks API key |
| `HOPSWORKS_PROJECT` | Your Hopsworks project name |

Add these **variables** (not secrets — they're non-sensitive):

| Variable | Example |
|---|---|
| `CITY_NAME` | `lahore` |
| `CITY_LAT` | `31.5204` |
| `CITY_LON` | `74.3587` |
| `AQICN_STATION` | `@7236` |

---

## 🌐 Deploying the Dashboard (Free)

1. Push your repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set **Main file path** to `app/dashboard.py`.
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   HOPSWORKS_API_KEY = "your_key"
   HOPSWORKS_PROJECT = "aqi_predictor"
   ```
5. Click **Deploy** — your dashboard is live at a free `.streamlit.app` URL.

---

## 📋 Final Submission Checklist

- [x] End-to-end AQI prediction system
- [x] Scalable, automated pipeline (GitHub Actions)
- [x] Interactive dashboard (Streamlit + Plotly)
- [x] SHAP feature importance explanations
- [x] Hazard alerts for dangerous AQI levels
- [x] Multiple models compared (Ridge, RF, GB, XGBoost)
- [x] EDA script with trend / correlation / time-pattern plots
- [x] Unit test suite
- [x] Full documentation (see `DOCUMENTATION.md`)

---

## 👤 Author

Shehryar — Final Year CS Project, University of Management and Technology (UMT), Lahore
