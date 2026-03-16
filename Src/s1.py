import yfinance as yf
import pandas as pd

# Download last 5 years of TCS stock data
tcs = yf.download("TCS.NS", period="5y")

# Save data to CSV file
tcs.to_csv("TCS_5year_stock.csv")

print("Download completed. CSV file created.")