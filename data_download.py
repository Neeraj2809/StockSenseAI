import yfinance as yf
import pandas as pd

ticker = input("Enter stock ticker: ")

data = yf.download(
    ticker,
    start="2015-01-01",
    end="2026-08-25",
    interval="1d",
    auto_adjust=False
)

if data.empty:
    print("Invalid ticker or no data found.")
else:
    data.to_csv("data/candlestick_data.csv")

    print("\nData downloaded successfully!")
    print("Stock:", ticker)
    print("Rows:", len(data))

    print("\nFirst 5 records:")
    print(data.head())