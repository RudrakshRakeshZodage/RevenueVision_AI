import os
import sys
import logging
import random
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import joblib

# Machine Learning models
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Deep Learning (TensorFlow)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Prophet
from prophet import Prophet

# Explainable AI
import shap

# Configure Logging and warnings
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seeds set to {seed}")

set_seeds(42)


def setup_directories():
    """Create models, plots, and exports directories if they don't exist."""
    for folder in ["models", "plots", "exports"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logging.info(f"Created directory: {folder}")


def load_and_clean_data(file_path):
    """Load and clean the raw Online Retail II dataset."""
    logging.info(f"Loading raw dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    logging.info(f"Raw shape: {df.shape}")
    
    # 1. Parse InvoiceDate
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    
    # 2. Filter cancellations and bad transactions
    initial_len = len(df)
    df = df[~df["Invoice"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    clean_len = len(df)
    logging.info(f"Filtered out {initial_len - clean_len} invalid transactions (cancellations/returns/bad data). {clean_len} remaining.")
    
    # 3. Drop exact duplicates
    df = df.drop_duplicates()
    logging.info(f"After duplicate removal, shape: {df.shape}")
    
    return df


def simulate_promotion_flag(date):
    """Simulate realistic e-commerce promotion campaigns based on date."""
    # Black Friday / Cyber Monday week: November 20 to November 30
    if date.month == 11 and date.day >= 20:
        return 1
    # Pre-Christmas shopping season: December 1 to December 15
    elif date.month == 12 and date.day <= 15:
        return 1
    # January Clearance: January 1 to January 15
    elif date.month == 1 and date.day <= 15:
        return 1
    # Spring Promo: April 1 to April 7
    elif date.month == 4 and date.day <= 7:
        return 1
    # Summer Clearance: July 1 to July 7
    elif date.month == 7 and date.day <= 7:
        return 1
    # Autumn Sale: October 1 to October 7
    elif date.month == 10 and date.day <= 7:
        return 1
    return 0


def aggregate_to_daily(df):
    """Aggregate transaction data into a daily time series."""
    logging.info("Aggregating transactions to daily dataset...")
    df["Date"] = df["InvoiceDate"].dt.date
    
    # Aggregate to daily metrics
    daily_df = df.groupby("Date").agg(
        Orders=("Invoice", "nunique"),
        Revenue=("Quantity", lambda x: np.sum(x * df.loc[x.index, "Price"])),
    ).reset_index()
    
    # Calculate Average Order Value (AOV)
    daily_df["Average_Order_Value"] = daily_df["Revenue"] / daily_df["Orders"]
    
    # Reindex to continuous daily frequency
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    daily_df.set_index("Date", inplace=True)
    
    full_date_range = pd.date_range(start=daily_df.index.min(), end=daily_df.index.max(), freq="D")
    daily_df = daily_df.reindex(full_date_range)
    daily_df.index.name = "Date"
    
    # Fill gaps
    daily_df["Orders"] = daily_df["Orders"].fillna(0.0).astype(float)
    daily_df["Revenue"] = daily_df["Revenue"].fillna(0.0)
    
    # Forward-fill Average Order Value (basket size remains similar), then back-fill/global-mean
    global_avg_aov = daily_df["Average_Order_Value"].mean()
    daily_df["Average_Order_Value"] = daily_df["Average_Order_Value"].ffill().fillna(global_avg_aov)
    
    daily_df = daily_df.reset_index()
    
    # Generate Holiday_Flag (UK Bank Holidays)
    uk_holidays = holidays.UnitedKingdom(years=list(daily_df["Date"].dt.year.unique()))
    daily_df["Holiday_Flag"] = daily_df["Date"].apply(lambda x: 1 if x in uk_holidays else 0)
    
    # Generate Promotion_Flag
    daily_df["Promotion_Flag"] = daily_df["Date"].apply(simulate_promotion_flag)
    
    # Generate basic date parts
    daily_df["Day_of_Week"] = daily_df["Date"].dt.dayofweek
    daily_df["Month"] = daily_df["Date"].dt.month
    daily_df["Quarter"] = daily_df["Date"].dt.quarter
    daily_df["Year"] = daily_df["Date"].dt.year
    
    logging.info(f"Aggregated daily shape: {daily_df.shape}")
    return daily_df


def engineer_features(df):
    """Engineers date, lag, rolling, and growth features."""
    logging.info("Engineering features for machine learning models...")
    df = df.copy()
    
    # 1. Date Features
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["year"] = df["Date"].dt.year
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)
    
    # 2. Lag Features on Orders
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df["Orders"].shift(lag)
        
    # 3. Rolling Features on Orders
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = df["Orders"].shift(1).rolling(window=window).mean()
        df[f"rolling_std_{window}"] = df["Orders"].shift(1).rolling(window=window).std()
        
    # 4. Business Features
    df["holiday_flag"] = df["Holiday_Flag"]
    df["promotion_flag"] = df["Promotion_Flag"]
    
    # 5. Growth Features on Orders
    eps = 1e-5
    df["daily_growth"] = (df["Orders"] - df["lag_1"]) / (df["lag_1"] + eps)
    df["weekly_growth"] = (df["Orders"] - df["lag_7"]) / (df["lag_7"] + eps)
    df["monthly_growth"] = (df["Orders"] - df["lag_30"]) / (df["lag_30"] + eps)
    
    # Clean up infinite/missing growth rates due to zero orders
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["daily_growth"] = df["daily_growth"].fillna(0.0)
    df["weekly_growth"] = df["weekly_growth"].fillna(0.0)
    df["monthly_growth"] = df["monthly_growth"].fillna(0.0)
    
    return df


def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics: MAE, RMSE, MAPE, R2 Score."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0.0
    r2 = r2_score(y_true, y_pred)
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2 Score": r2}


def create_lstm_sequences(X, y, seq_length=14):
    """Prepare inputs in 3D format [samples, time_steps, features] for LSTM."""
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)


def recursive_ml_forecast(model, model_name, train_df, forecast_dates, feature_cols, seq_length=14):
    """Generates forecasts recursively day-by-day for a forecast horizon."""
    full_df = train_df.copy()
    
    future_rows = pd.DataFrame(index=range(len(forecast_dates)))
    future_rows["Date"] = forecast_dates
    future_rows["Orders"] = 0.0  # Placeholder as float
    future_rows["Average_Order_Value"] = train_df["Average_Order_Value"].mean()
    
    uk_holidays = holidays.UnitedKingdom(years=list(forecast_dates.year.unique()))
    future_rows["Holiday_Flag"] = future_rows["Date"].apply(lambda x: 1 if x in uk_holidays else 0)
    future_rows["Holiday_Flag"] = future_rows["Holiday_Flag"].astype(int)
    future_rows["Promotion_Flag"] = future_rows["Date"].apply(simulate_promotion_flag)
    
    future_rows["day"] = future_rows["Date"].dt.day
    future_rows["month"] = future_rows["Date"].dt.month
    future_rows["quarter"] = future_rows["Date"].dt.quarter
    future_rows["year"] = future_rows["Date"].dt.year
    future_rows["week_of_year"] = future_rows["Date"].dt.isocalendar().week.astype(int)
    future_rows["day_of_week"] = future_rows["Date"].dt.dayofweek
    future_rows["is_weekend"] = future_rows["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)
    future_rows["holiday_flag"] = future_rows["Holiday_Flag"]
    future_rows["promotion_flag"] = future_rows["Promotion_Flag"]
    
    for col in feature_cols:
        if col not in future_rows.columns:
            future_rows[col] = 0.0
            
    full_df = pd.concat([full_df, future_rows], ignore_index=True)
    start_idx = len(train_df)
    
    for i in range(start_idx, len(full_df)):
        # Update lags
        for lag in [1, 7, 14, 30]:
            full_df.loc[i, f"lag_{lag}"] = full_df.loc[i - lag, "Orders"]
            
        # Update rolling features
        for window in [7, 14, 30]:
            hist_orders = full_df.loc[i - window : i - 1, "Orders"]
            full_df.loc[i, f"rolling_mean_{window}"] = hist_orders.mean()
            full_df.loc[i, f"rolling_std_{window}"] = hist_orders.std()
            
        # Update growth rates
        full_df.loc[i, "daily_growth"] = 0.0
        full_df.loc[i, "weekly_growth"] = 0.0
        full_df.loc[i, "monthly_growth"] = 0.0
        
        if model_name == "LSTM":
            seq_features = full_df.loc[i - seq_length : i - 1, feature_cols].values
            seq_features = np.expand_dims(seq_features, axis=0)
            pred = model.predict(seq_features, verbose=0)[0][0]
        else:
            X_pred = full_df.loc[[i], feature_cols]
            pred = model.predict(X_pred)[0]
            
        pred = max(0.0, pred)
        full_df.loc[i, "Orders"] = pred
        
    forecast_df = full_df.iloc[start_idx:].copy()
    return forecast_df["Orders"].values


def main():
    setup_directories()
    
    # 1. Load Data from proper path
    raw_df = load_and_clean_data("data/online_retail_II.csv")
    
    # 2. Aggregate
    daily_df = aggregate_to_daily(raw_df)
    
    # 3. Feature Engineering
    feat_df = engineer_features(daily_df)
    
    # Define features
    feature_cols = [
        "day", "month", "quarter", "year", "week_of_year", "day_of_week", "is_weekend",
        "holiday_flag", "promotion_flag",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_7", "rolling_std_14", "rolling_std_30",
        "daily_growth", "weekly_growth", "monthly_growth"
    ]
    target_col = "Orders"
    
    # Train-test split
    ml_df = feat_df.iloc[30:].reset_index(drop=True)
    train_size = int(len(ml_df) * 0.8)
    train_df = ml_df.iloc[:train_size].reset_index(drop=True)
    test_df = ml_df.iloc[train_size:].reset_index(drop=True)
    
    logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    model_metrics = {}
    test_predictions = pd.DataFrame({"Actual": y_test.values}, index=test_df["Date"])
    
    # ------------------ 1. Prophet ------------------
    logging.info("Training Facebook Prophet...")
    prophet_train = train_df[["Date", "Orders"]].rename(columns={"Date": "ds", "Orders": "y"})
    prophet_test = test_df[["Date", "Orders"]].rename(columns={"Date": "ds", "Orders": "y"})
    prophet_train["holiday_flag"] = train_df["holiday_flag"]
    prophet_train["promotion_flag"] = train_df["promotion_flag"]
    prophet_test["holiday_flag"] = test_df["holiday_flag"]
    prophet_test["promotion_flag"] = test_df["promotion_flag"]
    
    m_prophet = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m_prophet.add_regressor("holiday_flag")
    m_prophet.add_regressor("promotion_flag")
    m_prophet.fit(prophet_train)
    
    prophet_pred = m_prophet.predict(prophet_test[["ds", "holiday_flag", "promotion_flag"]])
    prophet_test_yhat = np.clip(prophet_pred["yhat"].values, 0, None)
    
    model_metrics["Prophet"] = calculate_metrics(y_test.values, prophet_test_yhat)
    test_predictions["Prophet"] = prophet_test_yhat
    
    # ------------------ 2. XGBoost ------------------
    logging.info("Training XGBoost...")
    xgb_reg = xgb.XGBRegressor(random_state=42)
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    xgb_grid = GridSearchCV(xgb_reg, xgb_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    
    xgb_test_pred = np.clip(best_xgb.predict(X_test), 0, None)
    model_metrics["XGBoost"] = calculate_metrics(y_test.values, xgb_test_pred)
    test_predictions["XGBoost"] = xgb_test_pred
    
    # ------------------ 3. LightGBM ------------------
    logging.info("Training LightGBM...")
    lgb_reg = lgb.LGBMRegressor(random_state=42, verbose=-1)
    lgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63]
    }
    lgb_grid = GridSearchCV(lgb_reg, lgb_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    lgb_grid.fit(X_train, y_train)
    best_lgb = lgb_grid.best_estimator_
    
    lgb_test_pred = np.clip(best_lgb.predict(X_test), 0, None)
    model_metrics["LightGBM"] = calculate_metrics(y_test.values, lgb_test_pred)
    test_predictions["LightGBM"] = lgb_test_pred
    
    # ------------------ 4. Random Forest ------------------
    logging.info("Training Random Forest...")
    rf_reg = RandomForestRegressor(random_state=42)
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(rf_reg, rf_params, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    
    rf_test_pred = np.clip(best_rf.predict(X_test), 0, None)
    model_metrics["RandomForest"] = calculate_metrics(y_test.values, rf_test_pred)
    test_predictions["RandomForest"] = rf_test_pred
    
    # ------------------ 5. LSTM ------------------
    logging.info("Training LSTM...")
    from sklearn.preprocessing import StandardScaler
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    
    seq_length = 14
    X_train_seq, y_train_seq = create_lstm_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_padded = np.vstack([X_train_scaled[-seq_length:], X_test_scaled])
    y_test_padded = np.concatenate([y_train_scaled[-seq_length:], scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()])
    X_test_seq, y_test_seq = create_lstm_sequences(X_test_padded, y_test_padded, seq_length)
    
    lstm_model = Sequential([
        Input(shape=(seq_length, len(feature_cols))),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    model_checkpoint = ModelCheckpoint(
        filepath='models/best_lstm_weights.keras',
        save_best_only=True,
        monitor='val_loss'
    )
    
    lstm_model.fit(
        X_train_seq, y_train_seq,
        epochs=100, batch_size=32, validation_split=0.2,
        callbacks=[early_stop, reduce_lr, model_checkpoint],
        verbose=0
    )
    
    lstm_test_pred_scaled = lstm_model.predict(X_test_seq, verbose=0)
    lstm_test_pred = scaler_y.inverse_transform(lstm_test_pred_scaled).flatten()
    lstm_test_pred = np.clip(lstm_test_pred, 0, None)
    
    model_metrics["LSTM"] = calculate_metrics(y_test.values, lstm_test_pred)
    test_predictions["LSTM"] = lstm_test_pred
    
    # ------------------ Metrics & Selection ------------------
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df.to_csv("exports/model_metrics.csv")
    logging.info(f"Saved exports/model_metrics.csv:\n{metrics_df}")
    
    best_model_name = metrics_df["MAPE"].idxmin()
    logging.info(f"Best model: {best_model_name}")
    
    # Save models
    joblib.dump(best_xgb, "models/xgboost_model.joblib")
    joblib.dump(best_lgb, "models/lightgbm_model.joblib")
    joblib.dump(best_rf, "models/random_forest.joblib")
    lstm_model.save("models/lstm_model.h5")
    with open("models/prophet_model.pkl", "wb") as f:
        pickle.dump(m_prophet, f)
        
    # Save generic best model
    if best_model_name == "XGBoost":
        joblib.dump(best_xgb, "models/best_model.joblib")
    elif best_model_name == "LightGBM":
        joblib.dump(best_lgb, "models/best_model.joblib")
    elif best_model_name == "RandomForest":
        joblib.dump(best_rf, "models/best_model.joblib")
    elif best_model_name == "Prophet":
        with open("models/best_model.pkl", "wb") as f:
            pickle.dump(m_prophet, f)
    elif best_model_name == "LSTM":
        lstm_model.save("models/best_model.h5")
        
    # ------------------ Forecasting ------------------
    last_hist_date = feat_df["Date"].max()
    forecast_dates_365 = pd.date_range(start=last_hist_date + timedelta(days=1), periods=365, freq="D")
    
    forecasts = {}
    uk_holidays = holidays.UnitedKingdom(years=list(forecast_dates_365.year.unique()))
    
    # 1. Prophet
    prophet_future = pd.DataFrame({"ds": forecast_dates_365})
    prophet_future["holiday_flag"] = prophet_future["ds"].apply(lambda x: 1 if x in uk_holidays else 0)
    prophet_future["promotion_flag"] = prophet_future["ds"].apply(simulate_promotion_flag)
    prophet_future_pred = m_prophet.predict(prophet_future)
    forecasts["Prophet"] = np.clip(prophet_future_pred["yhat"].values, 0, None)
    
    # 2. XGBoost
    forecasts["XGBoost"] = recursive_ml_forecast(best_xgb, "XGBoost", feat_df, forecast_dates_365, feature_cols)
    
    # 3. LightGBM
    forecasts["LightGBM"] = recursive_ml_forecast(best_lgb, "LightGBM", feat_df, forecast_dates_365, feature_cols)
    
    # 4. Random Forest
    forecasts["RandomForest"] = recursive_ml_forecast(best_rf, "RandomForest", feat_df, forecast_dates_365, feature_cols)
    
    # 5. LSTM
    class LSTMWrapper:
        def __init__(self, model, scaler_x, scaler_y):
            self.model = model
            self.scaler_x = scaler_x
            self.scaler_y = scaler_y
        def predict(self, X_seq_raw, *args, **kwargs):
            seq_len, num_feats = X_seq_raw.shape[1], X_seq_raw.shape[2]
            X_flat = X_seq_raw.reshape(-1, num_feats)
            X_scaled = self.scaler_x.transform(X_flat)
            X_seq_scaled = X_scaled.reshape(1, seq_len, num_feats)
            pred_scaled = self.model.predict(X_seq_scaled, verbose=0)
            pred = self.scaler_y.inverse_transform(pred_scaled).flatten()[0]
            return np.array([[pred]])
            
    lstm_wrapper = LSTMWrapper(lstm_model, scaler_x, scaler_y)
    forecasts["LSTM"] = recursive_ml_forecast(lstm_wrapper, "LSTM", feat_df, forecast_dates_365, feature_cols, seq_length)
    
    best_orders_forecast = forecasts[best_model_name]
    recent_aov = feat_df["Average_Order_Value"].iloc[-30:].mean()
    best_revenue_forecast = best_orders_forecast * recent_aov
    
    orders_forecast_df = pd.DataFrame({
        "Date": forecast_dates_365,
        "Orders_Expected": best_orders_forecast,
        "Orders_Best": best_orders_forecast * 1.20,
        "Orders_Worst": best_orders_forecast * 0.80,
    })
    
    revenue_forecast_df = pd.DataFrame({
        "Date": forecast_dates_365,
        "Revenue_Expected": best_revenue_forecast,
        "Revenue_Best": best_revenue_forecast * 1.20,
        "Revenue_Worst": best_revenue_forecast * 0.80,
    })
    
    full_forecast_df = pd.concat([orders_forecast_df, revenue_forecast_df.drop(columns=["Date"])], axis=1)
    
    # Save exports
    full_forecast_df.head(30).to_csv("exports/forecast_30_days.csv", index=False)
    full_forecast_df.head(90).to_csv("exports/forecast_90_days.csv", index=False)
    full_forecast_df.to_csv("exports/forecast_365_days.csv", index=False)
    revenue_forecast_df.to_csv("exports/revenue_forecast.csv", index=False)
    
    logging.info("CSVs exported to exports/")
    
    # Explainability (SHAP)
    explainer_model = best_lgb if "LightGBM" in [best_model_name, "LightGBM"] else best_xgb
    explainer_name = "LightGBM" if "LightGBM" in [best_model_name, "LightGBM"] else "XGBoost"
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer(X_train)
    
    # ------------------ Visualizations ------------------
    # 1. Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(test_predictions.index, test_predictions["Actual"], label="Actual Orders", color="royalblue", alpha=0.8)
    plt.plot(test_predictions.index, test_predictions[best_model_name], label=f"Predicted ({best_model_name})", color="crimson", linestyle="--")
    plt.title(f"Actual vs Predicted Orders - Test Set (Model: {best_model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Order Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/actual_vs_predicted.png", dpi=150)
    plt.close()
    
    # 2. Model Comparison
    plt.figure(figsize=(10, 5))
    sns.barplot(x=metrics_df.index, y=metrics_df["MAPE"], palette="viridis")
    plt.title("Model Comparison - Mean Absolute Percentage Error (MAPE)", fontsize=14, fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("MAPE (%)")
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=150)
    plt.close()
    
    # 3. Future Forecast Plot
    plt.figure(figsize=(14, 6))
    hist_to_plot = feat_df.tail(90)
    plt.plot(hist_to_plot["Date"], hist_to_plot["Orders"], label="Historical Orders", color="royalblue")
    plt.plot(full_forecast_df["Date"], full_forecast_df["Orders_Expected"], label=f"Forecast ({best_model_name})", color="crimson")
    plt.fill_between(full_forecast_df["Date"], full_forecast_df["Orders_Worst"], full_forecast_df["Orders_Best"], color="crimson", alpha=0.15, label="Operational Range")
    plt.title("E-Commerce Order Volume Demand Forecast (365 Days)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Order Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/future_forecast.png", dpi=150)
    plt.close()
    
    # 4. Revenue Forecast Plot
    plt.figure(figsize=(14, 6))
    hist_rev_to_plot = feat_df.tail(90)
    plt.plot(hist_rev_to_plot["Date"], hist_rev_to_plot["Revenue"], label="Historical Revenue", color="teal")
    plt.plot(full_forecast_df["Date"], full_forecast_df["Revenue_Expected"], label="Revenue Forecast", color="darkorange")
    plt.fill_between(full_forecast_df["Date"], full_forecast_df["Revenue_Worst"], full_forecast_df["Revenue_Best"], color="darkorange", alpha=0.15, label="Revenue Range")
    plt.title("E-Commerce Revenue Projection Forecast (365 Days)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Revenue ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/revenue_forecast.png", dpi=150)
    plt.close()
    
    # 5. Trend Plot
    plt.figure(figsize=(12, 5))
    plt.plot(feat_df["Date"], feat_df["rolling_mean_30"], label="30-day Rolling Mean Trend", color="darkviolet", linewidth=2)
    plt.title("E-Commerce Sales Growth Trend Analysis (30-day Rolling Average)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Trend Level (Orders)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/trend_analysis.png", dpi=150)
    plt.close()
    
    # 6. Seasonality Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg = feat_df.groupby("day_of_week")["Orders"].mean()
    sns.barplot(x=dow_avg.index, y=dow_avg.values, ax=axes[0], palette="Blues_d")
    axes[0].set_title("Weekly Seasonality (Avg Orders per Day of Week)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(dow_labels)
    
    month_avg = feat_df.groupby("month")["Orders"].mean()
    sns.barplot(x=month_avg.index, y=month_avg.values, ax=axes[1], palette="Oranges_d")
    axes[1].set_title("Monthly Seasonality (Avg Orders per Month)", fontsize=12, fontweight="bold")
    
    plt.suptitle("Seasonality Decomposition Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/seasonality.png", dpi=150)
    plt.close()
    
    # 7. Confidence Interval Plot
    plt.figure(figsize=(10, 5))
    forecast_short = full_forecast_df.head(90)
    plt.plot(forecast_short["Date"], forecast_short["Orders_Expected"], label="Expected Demand", color="indigo", linewidth=2)
    plt.fill_between(forecast_short["Date"], forecast_short["Orders_Worst"], forecast_short["Orders_Best"], color="indigo", alpha=0.2, label="Risk Band")
    plt.title("Operational Demand Planning & Confidence Intervals (Next 90 Days)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Order Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/confidence_intervals.png", dpi=150)
    plt.close()
    
    # 8. Feature Importance
    plt.figure(figsize=(12, 6))
    if hasattr(explainer_model, "feature_importances_"):
        importances = explainer_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        sns.barplot(x=importances[indices], y=[feature_cols[i] for i in indices], palette="rocket")
        plt.title(f"Top 15 Feature Importances ({explainer_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.close()
    
    # 9. SHAP Summary
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f"SHAP Feature Attribution Summary ({explainer_name})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/shap_summary.png", dpi=150)
    plt.close()
    
    logging.info("Plots saved in plots/")
    logging.info("Pipeline run complete.")


if __name__ == "__main__":
    main()
