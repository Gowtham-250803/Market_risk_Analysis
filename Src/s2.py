import yfinance as yf
import pandas as pd

# Download last 5 years of WIPRO stock data
tcs = yf.download("WIPRO.NS", period="5y")

# Save data to CSV file
tcs.to_csv("WIPRO_5year_stock.csv")

print("Download completed. CSV file created.")