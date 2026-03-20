import pandas as pd
import numpy as np
import os
import warnings


def _read_and_clean_price_file(file_path):
    """Read a CSV and normalize it to a clean numeric time series.

    Many download workflows add extra metadata rows (e.g., "Ticker" / "Date")
    or use a non-standard index name ("Price" instead of "Date").

    This helper handles:
    - stripping out non-date rows before parsing
    - coercing the index to datetime (day-first format)
    - converting all columns to numeric (non-numeric values become NaN)
    """

    df = pd.read_csv(file_path, index_col=0, dtype=str)

    # Some files contain extra header/metadata rows (e.g., "Ticker", "Date") in the first column.
    # Those rows will become NaT after coercion and are removed.
    original_row_count = len(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Suppress date parsing warnings
        df.index = pd.to_datetime(df.index, errors='coerce', dayfirst=True)
    df = df[~df.index.isna()]
    cleaned_row_count = len(df)

    if cleaned_row_count < original_row_count:
        print(f"Warning: {original_row_count - cleaned_row_count} non-date rows removed from {file_path} (likely metadata)")

    # Convert numeric columns to floats/ints; non-numeric items become NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    return df


def load_multiple_data(file_list):
    """Loads multiple CSV files and combines their Close prices into one DataFrame."""

    combined_prices = pd.DataFrame()

    for file_path in file_list:
        # Extract the asset name from the file name (e.g., 'AAPL.csv' becomes 'AAPL')
        asset_name = os.path.splitext(os.path.basename(file_path))[0]

        df = _read_and_clean_price_file(file_path)

        # Grab ONLY the 'Close' price and add it as a new column in our master table
        if 'Close' in df.columns:
            combined_prices[asset_name] = df['Close']
        else:
            print(f"Warning: 'Close' column not found in {file_path}")

    # Align on a common date index and forward-fill missing points before dropping any remaining NaNs.
    # This prevents a single missing row from dropping the entire dataset.
    combined_prices = combined_prices.sort_index()
    combined_prices = combined_prices.ffill().dropna()

    return combined_prices

def calculate_log_returns(prices_df):
    """Calculates daily logarithmic returns."""
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
    return log_returns