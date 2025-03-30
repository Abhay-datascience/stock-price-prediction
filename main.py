from flask import Flask, render_template, request
import joblib
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("stock_model.pkl")

# Ensure 'static' folder exists for storing images
if not os.path.exists("static"):
    os.makedirs("static")


# Function to fetch stock data and compute indicators
def get_stock_features(ticker):
    ticker = ticker.upper()

    # Append '.NS' for NSE stocks (default to NSE)
    if not ticker.endswith(('.NS', '.BO')):
        ticker += '.NS'

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="60d")

        if df.empty:
            return None, None

        # Create stock chart
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df["Close"], label="Close Price", color='blue')
        plt.title(f"{ticker} Stock Price (Last 60 Days)")
        plt.xlabel("Date")
        plt.ylabel("Close Price (INR)")
        plt.legend()
        plt.grid()

        # Save the plot
        chart_path = "static/stock_chart.png"
        plt.savefig(chart_path)
        plt.close()

        # Feature Engineering
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # RSI Calculation
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

        df['STD_20'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['SMA_20'] + (df['STD_20'] * 2)
        df['Lower_Band'] = df['SMA_20'] - (df['STD_20'] * 2)
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        df.dropna(inplace=True)
        latest_data = df.iloc[-1][['SMA_20', 'EMA_20', 'RSI_14', 'Upper_Band', 'Lower_Band', 'MACD', 'Signal_Line']]

        return np.array([latest_data.values]), chart_path

    except Exception as e:
        print("Error fetching data:", e)
        return None, None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker = request.form["ticker"].upper()
        input_data, chart_path = get_stock_features(ticker)

        if input_data is None:
            return render_template("index.html", error="Not enough data for analysis. Try another stock.")

        # Predict stock price
        predicted_price = model.predict(input_data)[0]
        predicted_price = round(float(predicted_price), 2)

        return render_template("result.html", ticker=ticker, predicted_price=predicted_price, chart_path=chart_path)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
