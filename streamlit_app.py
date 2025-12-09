# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, date
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV Date Viewer + Dec 2025 Forecast", layout="wide")

st.title("CSV Date viewer + Dec 2025 prediction (temperature + other params)")

# ---- Load CSV ----
st.markdown("**Load CSV** — either upload a file or the app will try `/mnt/data/DASH_comeback.csv` if available.")
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

@st.cache_data
def load_df(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        fallback = "/mnt/data/DASH_comeback.csv"
        if os.path.exists(fallback):
            df = pd.read_csv(fallback)
        else:
            st.error("No file uploaded and `/mnt/data/DASH_comeback.csv` not found. Upload a CSV to proceed.")
            return None
    return df

df = load_df(uploaded)

if df is None:
    st.stop()

st.write("**Preview (first 5 rows)**")
st.dataframe(df.head())

# ---- Try to find the datetime column ----
possible_date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
date_col = None

if possible_date_cols:
    date_col = st.selectbox("Select the datetime column detected (or choose another)", options=possible_date_cols + ["Other / None"])
    if date_col == "Other / None":
        date_col = st.selectbox("Choose any column as datetime", options=["-- pick --"] + list(df.columns))
        if date_col == "-- pick --":
            st.error("Please select a date/time column from your CSV.")
            st.stop()
else:
    date_col = st.selectbox("No obvious date column found. Choose the datetime column from the CSV", options=["-- pick --"] + list(df.columns))
    if date_col == "-- pick --":
        st.error("Please select a date/time column from your CSV.")
        st.stop()

# Try parse dates
def parse_dates_safe(df_, col):
    df = df_.copy()
    try:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    return df

df = parse_dates_safe(df, date_col)

if df[date_col].isna().all():
    st.error("Could not parse any dates from the selected column. Please preprocess your CSV or choose another column.")
    st.stop()

# Drop rows without parseable date
df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)

# Make sure index is datetime
df['date'] = df[date_col].dt.date
df['datetime'] = pd.to_datetime(df[date_col])

# Ask user to select parameter columns for display and prediction
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in CSV (required for plotting & prediction).")
    st.stop()

st.sidebar.header("Display & Prediction Settings")
st.sidebar.write("Numeric columns detected:")
selected_cols = st.sidebar.multiselect("Choose parameter(s) to display & predict (select at least one)", numeric_cols, default=[numeric_cols[0]])

# Year selector: only show previous years present in data
years_present = sorted(df['datetime'].dt.year.unique())
prev_years = [y for y in years_present if y < datetime.now().year]
if not prev_years:
    st.warning("No previous-year data found in file. The app will still try to predict using whatever historical data exists.")
    year_choice = st.sidebar.selectbox("Select year to display (if present)", options=years_present)
else:
    year_choice = st.sidebar.selectbox("Select previous year to display", options=prev_years)

# Filter to chosen year
df_year = df[df['datetime'].dt.year == int(year_choice)]

st.subheader(f"Rows for year {year_choice}")
st.dataframe(df_year.head(200))

# Basic plots for chosen year
st.subheader(f"Plots for {year_choice}")
for col in selected_cols:
    st.write(f"### {col}")
    if df_year.empty:
        st.write("No rows for selected year.")
        continue
    fig, ax = plt.subplots()
    ax.plot(df_year['datetime'], df_year[col], marker='.', linestyle='-')
    ax.set_xlabel("Date")
    ax.set_ylabel(col)
    ax.set_title(f"{col} — {year_choice}")
    st.pyplot(fig)

# -------------------------
# Forecast Dec 2025
# -------------------------
st.header("Forecast: December 2025 (daily)")

# Prepare single-target forecast for each selected column and show results
def prepare_features(ts):
    # ts: pandas Series indexed by datetime
    df_f = pd.DataFrame({'y': ts.values}, index=ts.index)
    df_f['dayofyear'] = df_f.index.dayofyear
    df_f['month'] = df_f.index.month
    df_f['day'] = df_f.index.day
    df_f['year'] = df_f.index.year
    # lag features (last 7 days)
    for lag in range(1, 8):
        df_f[f'lag_{lag}'] = df_f['y'].shift(lag)
    # rolling mean
    df_f['rmean_7'] = df_f['y'].rolling(7, min_periods=1).mean().shift(1)
    df_f = df_f.dropna()
    return df_f

def train_and_predict_single(ts_series, predict_dates):
    # ts_series: pandas Series with datetime index
    df_feat = prepare_features(ts_series)
    if df_feat.shape[0] < 30:
        # Not enough rows; build simpler features from raw index (no lags)
        X = pd.DataFrame({
            'dayofyear': df_feat.index.dayofyear,
            'month': df_feat.index.month,
            'year': df_feat.index.year
        }, index=df_feat.index)
        y = df_feat['y']
    else:
        X = df_feat.drop(columns=['y'])
        y = df_feat['y']

    # Train-test split via timeseries split
    tscv = TimeSeriesSplit(n_splits=3)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    # Fit on all data
    model.fit(X, y)

    # Build features for predict_dates
    pred_index = pd.to_datetime(predict_dates)
    X_pred = pd.DataFrame(index=pred_index)
    X_pred['dayofyear'] = X_pred.index.dayofyear
    X_pred['month'] = X_pred.index.month
    X_pred['day'] = X_pred.index.day
    X_pred['year'] = X_pred.index.year

    # For lag/rolling features we will use last known values from ts_series
    # create temporary series that extends historical series with NaNs for predict dates
    combined_index = ts_series.index.union(pred_index)
    combined = pd.Series(index=combined_index, dtype=float)
    combined.loc[ts_series.index] = ts_series.values

    # compute lag features for prediction rows using last known values (recursive forecasting)
    preds = []
    history = combined.copy()
    for dt in pred_index:
        # prepare features for current dt
        dayofyear = dt.timetuple().tm_yday
        month = dt.month
        day = dt.day
        yearv = dt.year
        # lags: take previous days from history (which will include earlier preds)
        lags = []
        for lag in range(1, 8):
            prev = dt - pd.Timedelta(days=lag)
            val = history.get(prev, np.nan)
            lags.append(val)
        rmean_7 = pd.Series([history.get(dt - pd.Timedelta(days=i), np.nan) for i in range(1,8)]).mean()
        feat = {
            'dayofyear': dayofyear,
            'month': month,
            'day': day,
            'year': yearv,
            'rmean_7': rmean_7
        }
        for i, lag_val in enumerate(lags, start=1):
            feat[f'lag_{i}'] = lag_val

        # If any lag is NaN and not enough history, fall back to simple features
        if np.isnan(list(feat.values())).any() and X.shape[1] < 6:
            feat_small = {'dayofyear': dayofyear, 'month': month, 'year': yearv}
            # align columns
            cols_needed = X.columns
            x_row = pd.DataFrame([feat_small], index=[dt])[cols_needed]
        else:
            cols_needed = X.columns
            x_row = pd.DataFrame([feat], index=[dt])[cols_needed]

        # Predict
        pred_val = model.predict(x_row)[0]
        preds.append(pred_val)
        # place it into history for future lag computation
        history.loc[dt] = pred_val

    pred_series = pd.Series(preds, index=pred_index, name='prediction')
    return pred_series

# Build predict dates: all days of Dec 2025
dec2025_dates = pd.date_range(start="2025-12-01", end="2025-12-31", freq='D')

all_predictions = {}
for col in selected_cols:
    st.write(f"Predicting **{col}** ...")
    ts = df.set_index('datetime')[col].dropna()
    if ts.empty:
        st.write("No historical numeric data for this column — skipping.")
        continue
    try:
        pred = train_and_predict_single(ts, dec2025_dates)
        all_predictions[col] = pred
        # show plot historical last year vs predicted Dec 2025
        fig, ax = plt.subplots(figsize=(10,4))
        # historical: last 180 days for context
        last_hist = ts.last('180D')
        ax.plot(last_hist.index, last_hist.values, label='historical (last 180 days)')
        ax.plot(pred.index, pred.values, label='pred Dec 2025', marker='o', linestyle='--')
        ax.set_title(f"{col}: historical vs Dec 2025 prediction")
        ax.legend()
        st.pyplot(fig)

        # show prediction table
        df_pred = pred.rename(col).reset_index()
        df_pred.columns = ['datetime', col]
        st.dataframe(df_pred)
    except Exception as e:
        st.error(f"Failed to train/predict for {col}: {e}")

# Download predictions combined as CSV
if all_predictions:
    combined_pred_df = pd.DataFrame(all_predictions)
    combined_pred_df.index.name = 'datetime'
    st.subheader("Combined predictions (Dec 2025)")
    st.dataframe(combined_pred_df)

    csv = combined_pred_df.reset_index().to_csv(index=False)
    st.download_button("Download predictions CSV", csv, "dec2025_predictions.csv", "text/csv")
else:
    st.info("No predictions were produced (maybe no numeric columns selected or insufficient data).")

st.write("---")
st.write("Notes:")
st.markdown("""- This app uses a **RandomForestRegressor** with simple lag and calendar features.  
- It's a fast, general-purpose forecasting method. For more accurate weather forecasts consider using domain models (prophet, LSTM, or physical-model outputs).  
- If your CSV uses different date formatting or timezone, pre-process the CSV so the date/time column parses correctly.  
- Want monthly totals instead of daily forecasts, or multi-output deep-learning models? Ask and I'll extend the app.
""")
