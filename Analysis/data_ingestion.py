import pandas as pd
import numpy as np

def load_data(filepath):
    """Loads the raw asset price data."""
    # Assuming a CSV with dates as the index and tickers as columns
    prices_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return prices_df

def calculate_log_returns(prices_df):
    """Calculates daily logarithmic returns."""
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
    return log_returns