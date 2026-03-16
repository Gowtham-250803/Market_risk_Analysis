import yfinance as yf
import pandas as pd

# Download last 5 years of IRFC stock data
tcs = yf.download("IRFC.NS", period="5y")

# Save data to CSV file
tcs.to_csv("IRFC_5year_stock.csv")

print("Download completed. CSV file created.")