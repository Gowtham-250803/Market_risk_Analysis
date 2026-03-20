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
    historical_values = np.asarray(historical_values).squeeze()

    # Compute a single time series of portfolio volatility.
    # If rolling_vol_30d is a DataFrame (asset-level), aggregate using the portfolio weights.
    if isinstance(rolling_vol_30d, pd.DataFrame):
        portfolio_returns = pd.Series(historical_port_returns, index=log_returns.index)
        rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252)
    else:
        # Assume it's already a 1-D series/array aligned with log_returns index
        rolling_vol = pd.Series(np.asarray(rolling_vol_30d).squeeze(), index=log_returns.index)

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

def export_efficient_frontier(results, weights_record, asset_names, filename='alphapulse_efficient_frontier.csv'):
    """
    Packages the Efficient Frontier simulations into a Tableau-ready CSV.
    Tags the 'Optimal' (Max Sharpe) and 'Minimum Risk' portfolios.
    """
    # Create a DataFrame from the results array
    ef_df = pd.DataFrame({
        'Expected_Return': results[0,:],
        'Risk_Volatility': results[1,:],
        'Sharpe_Ratio': results[2,:]
    })
    
    # Add the weights for each asset as separate columns
    for counter, asset in enumerate(asset_names):
        ef_df[f'Weight_{asset}'] = [w[counter] for w in weights_record]
        
    # Find the Optimal and Min Risk portfolios to flag them for Tableau
    max_sharpe_idx = ef_df['Sharpe_Ratio'].idxmax()
    min_vol_idx = ef_df['Risk_Volatility'].idxmin()
    
    # Create a 'Portfolio_Type' column for easy color-coding in Tableau
    ef_df['Portfolio_Type'] = 'Simulated Portfolio'
    ef_df.loc[max_sharpe_idx, 'Portfolio_Type'] = 'Max Sharpe (Optimal)'
    ef_df.loc[min_vol_idx, 'Portfolio_Type'] = 'Minimum Volatility'
    
    # Export to CSV
    ef_df.to_csv(filename, index=False)
    print(f"Exported Efficient Frontier with {len(ef_df)} simulations: {filename}")