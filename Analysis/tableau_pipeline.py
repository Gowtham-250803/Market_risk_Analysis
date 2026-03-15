import os
import pandas as pd
import numpy as np

def export_correlation_matrix(log_returns, output_dir='Output', filename='alphapulse_correlation_matrix.csv'):
    """Melts the correlation matrix into an edge-list for Tableau heatmaps."""
    corr_matrix = log_returns.corr()
    corr_melted = corr_matrix.reset_index().melt(id_vars='index')
    corr_melted.columns = ['Asset_1', 'Asset_2', 'Correlation']
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    corr_melted.to_csv(filepath, index=False)
    print(f"Exported: {filepath}")

def export_monte_carlo_bands(portfolio_paths, time_horizon=252, output_dir='Output', filename='alphapulse_mc_bands.csv'):
    """Extracts daily percentiles to create a lightweight 'Cone of Uncertainty'."""
    percentiles = np.percentile(portfolio_paths, [5, 25, 50, 75, 95], axis=0)
    
    mc_bands_df = pd.DataFrame({
        'Trading_Day': np.arange(1, time_horizon + 1),
        'P5_WorstCase': percentiles[0],
        'P25_LowerBand': percentiles[1],
        'P50_Median': percentiles[2],
        'P75_UpperBand': percentiles[3],
        'P95_BestCase': percentiles[4]
    })
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    mc_bands_df.to_csv(filepath, index=False)
    print(f"Exported: {filepath}")

def export_historical_metrics(log_returns, rolling_vol_30d, weights, initial_value=1000000, output_dir='Output', filename='alphapulse_historical_risk.csv'):
    """Calculates daily drawdowns, risk regimes, and historical portfolio values."""
    # Reconstruct historical portfolio daily values
    historical_port_returns = np.dot(log_returns, weights)
    cumulative_hist_returns = np.exp(np.cumsum(historical_port_returns))
    historical_values = initial_value * cumulative_hist_returns
    # Ensure 1-D arrays/series for DataFrame construction
    historical_values = np.asarray(historical_values).squeeze()
    if hasattr(rolling_vol_30d, 'values'):
        rolling_vol = np.asarray(rolling_vol_30d).squeeze()
    else:
        rolling_vol = np.asarray(rolling_vol_30d).squeeze()

    historical_data = pd.DataFrame({
        'Date': log_returns.index,
        'Portfolio_Value': historical_values,
        'Rolling_Vol_30d': rolling_vol
    })
    
    # Calculate Daily Drawdown
    rolling_peak = historical_data['Portfolio_Value'].cummax()
    historical_data['Drawdown_Pct'] = (historical_data['Portfolio_Value'] - rolling_peak) / rolling_peak
    
    # Flag Risk Regimes (80th percentile threshold)
    high_vol_threshold = historical_data['Rolling_Vol_30d'].quantile(0.80)
    historical_data['Risk_Regime'] = np.where(
        historical_data['Rolling_Vol_30d'] > high_vol_threshold, 
        'High Volatility', 
        'Normal'
    )
    
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    historical_data.dropna().to_csv(filepath, index=False)
    print(f"Exported: {filepath}")