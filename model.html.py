import yfinance as yf
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import yfinance as yf

# Download historical stock data (e.g., MRF)
df = yf.download('MRF.NS', start='2020-01-01', end='2025-01-01')
df.head()
# Load stock data
df = yf.download('AAPL', start='2022-01-01', end='2025-01-01')

# SMA & EMA
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI_14'] = calculate_rsi(df, 14)

# Bollinger Bands
df['STD_20'] = df['Close'].rolling(window=20).std()
df['Upper_Band'] = df['SMA_20'] + (df['STD_20'] * 2)
df['Lower_Band'] = df['SMA_20'] - (df['STD_20'] * 2)

# MACD
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

df.fillna(method= 'bfill',inplace= True)

x= df[['SMA_20', 'EMA_20', 'RSI_14', 'Upper_Band', 'Lower_Band', 'MACD', 'Signal_Line']]
y= df['Close']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size= 0.2,random_state= 54)

model= LinearRegression()
model.fit(x_train,y_train)
y_pred= model.predict(x_test)

joblib.dump(model, "stock_model.pkl")
print("Model trained and saved as 'stock_model.pkl'")
