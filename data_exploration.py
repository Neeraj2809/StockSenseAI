import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

def calculate_rsi(data, period=14):
    delta = data["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


df = pd.read_csv("data/candlestick_data.csv", skiprows=[1, 2])

df = df.rename(columns={"Price": "Date"})

df["Date"] = pd.to_datetime(df["Date"])

df = df[[
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]]

df = df.sort_values("Date")


# Feature Engineering

df["Daily_Return"] = df["Close"].pct_change() * 100

df["Price_Change"] = df["Close"].diff()

df["Daily_Range"] = df["High"] - df["Low"]

df["MA_10"] = df["Close"].rolling(10).mean()

df["MA_20"] = df["Close"].rolling(20).mean()

df["MA_50"] = df["Close"].rolling(50).mean()

df["MA_200"] = df["Close"].rolling(200).mean()

df["Above_MA_50"] = (
    df["Close"] > df["MA_50"]
).astype(int)


# RSI

df["RSI"] = calculate_rsi(df)


# MACD

ema_12 = df["Close"].ewm(
    span=12,
    adjust=False
).mean()

ema_26 = df["Close"].ewm(
    span=26,
    adjust=False
).mean()

df["MACD"] = ema_12 - ema_26

df["MACD_Signal"] = df["MACD"].ewm(
    span=9,
    adjust=False
).mean()

df["MACD_Histogram"] = (
    df["MACD"] - df["MACD_Signal"]
)


# Bollinger Bands

df["BB_Middle"] = df["Close"].rolling(20).mean()

bb_std = df["Close"].rolling(20).std()

df["BB_Upper"] = (
    df["BB_Middle"] + (2 * bb_std)
)

df["BB_Lower"] = (
    df["BB_Middle"] - (2 * bb_std)
)

# Volatility and Volume

df["Volatility_20"] = (
    df["Daily_Return"].rolling(20).std()
)

df["Volume_Change"] = (
    df["Volume"].pct_change()
    .replace([float("inf"), float("-inf")], 0)
    * 100
)

# Price Range Features

df["High_Low_Percent"] = (
    (df["High"] - df["Low"]) /
    df["Close"]
) * 100

df["Close_Open_Percent"] = (
    (df["Close"] - df["Open"]) /
    df["Open"]
) * 100


# Create Target

df["Next_Close"] = df["Close"].shift(-1)

df["Target"] = (
    df["Next_Close"] > df["Close"]
).astype(int)

df = df.replace([float("inf"), float("-inf")], pd.NA)
df = df.dropna()
# Select Features for Machine Learning
features = [
    "Daily_Return",
    "Price_Change",
    "Daily_Range",
    "MA_10",
    "MA_20",
    "MA_50",
    "MA_200",
    "Above_MA_50",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Middle",
    "BB_Upper",
    "BB_Lower",
    "Volatility_20",
    "Volume_Change",
    "High_Low_Percent",
    "Close_Open_Percent"
]

X = df[features]

y_classification = df["Target"]
y_regression = df["Next_Close"]

y = y_classification

# Time-Based Train Test Split

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

y_reg_train = y_regression.iloc[:split]
y_reg_test = y_regression.iloc[split:]


print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# Feature Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# SVM Model

svm_model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

svm_model.fit(
    X_train_scaled,
    y_train
)


# Prediction

svm_prediction = svm_model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    svm_prediction
)

precision = precision_score(
    y_test,
    svm_prediction
)

recall = recall_score(
    y_test,
    svm_prediction
)

f1 = f1_score(
    y_test,
    svm_prediction
)

cm = confusion_matrix(
    y_test,
    svm_prediction
)


print("\nSVM Results")
print("--------------------")

print("Accuracy:",
      round(accuracy * 100, 2), "%")

print("Precision:",
      round(precision * 100, 2), "%")

print("Recall:",
      round(recall * 100, 2), "%")

print("F1 Score:",
      round(f1 * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    svm_prediction
))

# XGBoost Model

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train,
    y_train
)

# Feature Correlation

correlation = X.corr()

print("\nFeature Correlation:")
print(correlation.round(2))

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

xgb_prediction = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_prediction
)

xgb_precision = precision_score(
    y_test,
    xgb_prediction
)

xgb_recall = recall_score(
    y_test,
    xgb_prediction
)

xgb_f1 = f1_score(
    y_test,
    xgb_prediction
)

xgb_cm = confusion_matrix(
    y_test,
    xgb_prediction
)

print("\nXGBoost Results")
print("--------------------")

print("Accuracy:",
      round(xgb_accuracy * 100, 2), "%")

print("Precision:",
      round(xgb_precision * 100, 2), "%")

print("Recall:",
      round(xgb_recall * 100, 2), "%")

print("F1 Score:",
      round(xgb_f1 * 100, 2), "%")

print("\nConfusion Matrix:")
print(xgb_cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    xgb_prediction
))

# XGBoost Regression Model

xgb_reg_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    objective="reg:squarederror"
)

xgb_reg_model.fit(X_train, y_reg_train)

reg_prediction = xgb_reg_model.predict(X_test)

mae = mean_absolute_error(y_reg_test, reg_prediction)
mse = mean_squared_error(y_reg_test, reg_prediction)
rmse = mse ** 0.5
r2 = r2_score(y_reg_test, reg_prediction)

print("\nXGBoost Regression Results")
print("----------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

print("\nFinal Dataset:")
print(df.head())

print("\nFinal Shape:")
print(df.shape)

print("\nTarget Distribution:")
print(df["Target"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())
