import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Weather Dashboard + AI Forecast", layout="wide")

st.title("📊 Weather Dashboard — Historical (CSV) + ML Forecast")

# -------------------------
# Config - change if needed
# -------------------------
CSV_PATH = "/mnt/data/DASH_comeback.csv"   # <- file uploaded on your environment
DATE_COLUMN = "date"                      # name of the date column in your CSV
TARGET_COLUMN = "temperature"             # name of the column to predict (change if needed)
N_LAGS = 14                               # number of lag days used as features
RANDOM_STATE = 42
# -------------------------

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # make a copy
    df = df.copy()

    # Ensure date column is parsed
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"CSV must contain a '{DATE_COLUMN}' column")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    # Sort by date
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # If target not in columns, try to guess a numeric column
    if TARGET_COLUMN not in df.columns:
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found to predict. Please set TARGET_COLUMN correctly.")
        else:
            st.warning(f"Target column '{TARGET_COLUMN}' not found. Using '{numeric_cols[0]}' instead.")
            global TARGET_COLUMN
            TARGET_COLUMN = numeric_cols[0]

    # forward fill / interpolate small gaps
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df[TARGET_COLUMN] = df[TARGET_COLUMN].interpolate().ffill().bfill()

    # return
    return df[[DATE_COLUMN, TARGET_COLUMN]]

def create_lag_features(series, n_lags):
    """Create a DataFrame with lag features for a univariate series."""
    df = pd.DataFrame({ 'y': series.values })
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df['y'].shift(lag)
    # optionally add rolling stats
    df['rolling_mean_3'] = df['y'].rolling(window=3, min_periods=1).mean().shift(1)
    df['rolling_std_7'] = df['y'].rolling(window=7, min_periods=1).std().shift(1).fillna(0)
    df = df.dropna().reset_index(drop=True)
    return df

@st.cache_data
def train_model(X, y):
    # Train-test split (keep last portion as test to simulate time)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))

    return model, rmse_train, rmse_test, X_train, X_test, y_train, y_test

def forecast_future(last_window, model, n_steps):
    """Given the last N lags (array), iteratively predict next n_steps."""
    preds = []
    window = list(last_window)  # most recent lags: lag_1, lag_2, ...
    for step in range(n_steps):
        # Build input vector identical to training feature order
        x = []
        for lag in range(1, N_LAGS + 1):
            # lag_1 corresponds to window[0]
            if lag <= len(window):
                x.append(window[-lag])  # note: window[-1] is latest value
            else:
                x.append(window[0])
        # Additional features: rolling stats pass previous last values:
        rolling_mean_3 = np.mean(window[-3:]) if len(window) >= 3 else np.mean(window)
        rolling_std_7 = np.std(window[-7:]) if len(window) >= 7 else np.std(window)
        x.append(rolling_mean_3)
        x.append(rolling_std_7)

        x_arr = np.array(x).reshape(1, -1)
        yhat = model.predict(x_arr)[0]
        preds.append(yhat)

        # update window: append yhat to window values and drop oldest
        window.append(yhat)
        if len(window) > N_LAGS:
            window.pop(0)

    return np.array(preds)

# -------------------------
# App UI: load data
# -------------------------
df_raw = load_csv(CSV_PATH)
if df_raw is None:
    st.error(f"CSV not found at {CSV_PATH}. Please upload your CSV to the server or update CSV_PATH.")
    st.stop()

st.subheader("Preview raw data")
st.write("First 5 rows from your CSV (auto-detected date and target column).")
st.dataframe(df_raw.head())

# -------------------------
# Preprocess
# -------------------------
try:
    df = preprocess(df_raw)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

st.subheader("Cleaned time series")
st.write(f"Date range: {df[DATE_COLUMN].min().date()} → {df[DATE_COLUMN].max().date()}")
st.line_chart(df.set_index(DATE_COLUMN)[TARGET_COLUMN])

# -------------------------
# Controls
# -------------------------
st.sidebar.header("Model & Forecast options")
n_lags = st.sidebar.slider("Number of lag features (days)", min_value=3, max_value=30, value=N_LAGS, step=1)
forecast_days = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=14)
train_button = st.sidebar.button("Train model & Forecast")

# Reassign chosen n_lags
N_LAGS = n_lags

# -------------------------
# Build lag features
# -------------------------
series = df[TARGET_COLUMN].reset_index(drop=True)
lagged = create_lag_features(series, N_LAGS)
# align dates (lagged rows correspond to dates starting from index N_LAGS)
dates_for_model = df[DATE_COLUMN].iloc[N_LAGS:].reset_index(drop=True)

st.write(f"Using {N_LAGS} lag features. Data points available for modeling: {len(lagged)}")

# -------------------------
# Train model when user clicks
# -------------------------
if train_button:
    with st.spinner("Training model..."):
        X = lagged.drop(columns=['y'])
        y = lagged['y'].values
        model, rmse_train, rmse_test, X_train, X_test, y_train, y_test = train_model(X.values, y)

    st.success("Model trained ✅")
    st.write(f"Train RMSE: {rmse_train:.3f} | Test RMSE: {rmse_test:.3f}")

    # Show test vs predicted
    test_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(y_test)), y_test, label="Actual (test)", marker='o', alpha=0.8)
    ax.plot(range(len(y_test)), test_pred, label="Predicted (test)", marker='x', alpha=0.8)
    ax.legend()
    ax.set_title("Test set: actual vs predicted")
    st.pyplot(fig)

    # Forecast future
    last_values = series.values[-N_LAGS:].tolist()
    future_preds = forecast_future(last_values, model, forecast_days)

    # Build future dates
    last_date = df[DATE_COLUMN].max()
    future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

    # Display forecast
    forecast_df = pd.DataFrame({ DATE_COLUMN: future_dates, f"pred_{TARGET_COLUMN}": future_preds })
    st.subheader(f"{forecast_days}-day Forecast for {TARGET_COLUMN}")
    st.dataframe(forecast_df)

    # Plot history vs forecast
    hist_plot = df.set_index(DATE_COLUMN)[TARGET_COLUMN].rename("history")
    combined = pd.concat([hist_plot, forecast_df.set_index(DATE_COLUMN)[f"pred_{TARGET_COLUMN}"]])
    st.line_chart(combined)

    # Optionally export forecast as CSV
    csv_export = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download forecast CSV", csv_export, file_name=f"forecast_{TARGET_COLUMN}.csv", mime="text/csv")

else:
    st.info("Press 'Train model & Forecast' in the left sidebar to train an ML model and predict future values.")

# -------------------------
# Quick summary / stats
# -------------------------
st.sidebar.header("Quick stats")
st.sidebar.write(f"Records in CSV: {len(df)}")
st.sidebar.write(f"Date range: {df[DATE_COLUMN].min().date()} → {df[DATE_COLUMN].max().date()}")
st.sidebar.write(f"Target column: {TARGET_COLUMN}")
st.sidebar.write("Model: RandomForest using lag features")

# Show monthly averages
st.subheader("Monthly averages (by calendar month)")
temp_monthly = df.set_index(DATE_COLUMN).resample("M").mean()[TARGET_COLUMN]
st.bar_chart(temp_monthly)

