"""Data preparation and model utilities for the StockSenseAI dashboard."""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

# Company	Yahoo Finance ticker
# Reliance Industries	RELIANCE.NS
# Tata Motors	TATAMOTORS.NS
# Infosys	INFY.NS
# TCS	TCS.NS
# State Bank of India	SBIN.NS
# ITC	ITC.NS
# HDFC Bank	HDFCBANK.NS
# Adani Green Energy	ADANIGREEN.NS
TICKER = "INFY.NS"

FEATURES = [
    "Daily_Return", "Price_Change", "Daily_Range", "MA_10", "MA_20",
    "MA_50", "MA_200", "Above_MA_50", "RSI", "MACD", "MACD_Signal",
    "MACD_Histogram", "BB_Middle", "BB_Upper", "BB_Lower", "Volatility_20",
    "Volume_Change", "High_Low_Percent", "Close_Open_Percent",
]


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def load_and_prepare_data() -> pd.DataFrame:
    """Download the configured ticker and reproduce the existing V1 feature preparation."""
    data = yf.download(
        TICKER,
        start="2015-01-01",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No market data was returned for {TICKER}.")

    # yfinance can return MultiIndex columns even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df = df.sort_values("Date")

    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Price_Change"] = df["Close"].diff()
    df["Daily_Range"] = df["High"] - df["Low"]
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()
    df["Above_MA_50"] = (df["Close"] > df["MA_50"]).astype(int)
    df["RSI"] = calculate_rsi(df)

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (2 * bb_std)
    df["BB_Lower"] = df["BB_Middle"] - (2 * bb_std)
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
    df["Volume_Change"] = (
        df["Volume"].pct_change().replace([float("inf"), float("-inf")], 0) * 100
    )
    df["High_Low_Percent"] = ((df["High"] - df["Low"]) / df["Close"]) * 100
    df["Close_Open_Percent"] = ((df["Close"] - df["Open"]) / df["Open"]) * 100
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove rows where technical indicators are not yet available.
    df = df.dropna(subset=FEATURES)

    return df


def choose_threshold(y_true, probabilities, target_precision=0.58):
    best_threshold = 0.50
    best_f1 = -1.0
    found_target = False

    for threshold in np.arange(0.40, 0.81, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )
        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        # Prefer thresholds meeting the precision target.
        if precision >= target_precision:
            if not found_target or f1 > best_f1:
                best_threshold = float(threshold)
                best_f1 = f1
                found_target = True

        # If no threshold meets the target, retain the
        # threshold with the best F1 score.
        elif not found_target and f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = f1

    return best_threshold

def train_models(df: pd.DataFrame) -> dict:
    """Train the existing models with the same chronological 80/20 split."""
    # Use only rows with known targets for model training.
    train_df = df.dropna(subset=["Target", "Next_Close"]).copy()

    X = train_df[FEATURES]
    y_classification = train_df["Target"].astype(int)
    y_regression = train_df["Next_Close"]

    split = int(len(train_df) * 0.8)
    if X.isna().any().any():
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values in features: {missing}")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_classification.iloc[:split], y_classification.iloc[split:]
    y_reg_train, y_reg_test = y_regression.iloc[:split], y_regression.iloc[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model = SVC(kernel="rbf", probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    xgb_model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        random_state=42, eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    xgb_reg_model = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        random_state=42, objective="reg:squarederror",
    )
    xgb_reg_model.fit(X_train, y_reg_train)

    svm_prediction = svm_model.predict(X_test_scaled)
    xgb_prediction = xgb_model.predict(X_test)
    reg_prediction = xgb_reg_model.predict(X_test)

    mse = mean_squared_error(y_reg_test, reg_prediction)

    # Step 2: Check that all test data lengths match
    print("Test dates:", len(train_df["Date"].iloc[split:]))
    print("Actual prices:", len(y_reg_test))
    print("Predicted prices:", len(reg_prediction))

    return {
        # Keep your existing return dictionary here
        "scaler": scaler,
        "svm_model": svm_model,
        "xgb_model": xgb_model,
        "xgb_reg_model": xgb_reg_model,
        "metrics": {
            "svm": _classification_metrics(y_test, svm_prediction),
            "xgb": _classification_metrics(y_test, xgb_prediction),
            "regression": {
                "MAE": mean_absolute_error(y_reg_test, reg_prediction),
                "MSE": mse,
                "RMSE": mse ** 0.5,
                "R2": r2_score(y_reg_test, reg_prediction),
            },
        },
        "test_dates": train_df["Date"].iloc[split:].reset_index(drop=True),
        "actual_close": y_reg_test.reset_index(drop=True),
        "predicted_close": pd.Series(reg_prediction).reset_index(drop=True),
    }


def _classification_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
    }


def predict_latest(models: dict, row: pd.DataFrame) -> dict:
    """Return predictions for one prepared feature row."""
    X_latest = row[FEATURES]
    svm_value = int(models["svm_model"].predict(models["scaler"].transform(X_latest))[0])
    xgb_value = int(models["xgb_model"].predict(X_latest)[0])
    next_close = float(models["xgb_reg_model"].predict(X_latest)[0])
    return {"svm": svm_value, "xgb": xgb_value, "next_close": next_close}
