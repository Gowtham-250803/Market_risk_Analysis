import os
import numpy as np
from data_ingestion import load_data, calculate_log_returns
from quant_engine import calculate_core_metrics, run_monte_carlo, calculate_var
from tableau_pipeline import export_correlation_matrix, export_monte_carlo_bands, export_historical_metrics

def run_alphapulse():
    print("Initializing AlphaPulse Quantitative Engine...")
    
    # Output directory for exports
    output_dir = 'Output'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Data Ingestion
    # Load cleaned price data (use Close prices for returns)
    prices = load_data('Data/Cleaned Data/clean_data.csv')
    if 'Close' in prices.columns:
        prices = prices[['Close']]
    log_returns = calculate_log_returns(prices)
    
    # 2. Quant Math Setup
    num_assets = len(prices.columns)
    weights = np.full(num_assets, 1/num_assets) # Equal weighting
    initial_value = 1000000 # $1M baseline
    
    mean_returns, cov_matrix, rolling_vol_30d = calculate_core_metrics(log_returns)
    
    # 3. Monte Carlo & VaR
    print("Running 10,000 Monte Carlo Simulations...")
    mc_paths = run_monte_carlo(mean_returns, cov_matrix, weights, initial_value)
    
    var_95 = calculate_var(mc_paths, initial_value)
    print(f"Statistical Sanity Check -> 95% VaR: ${var_95:,.2f}")
    
    # 4. Tableau Data Engineering & Export
    print("Formatting data for Tableau visualization...")
    export_correlation_matrix(log_returns, output_dir=output_dir)
    export_monte_carlo_bands(mc_paths, output_dir=output_dir)
    export_historical_metrics(log_returns, rolling_vol_30d, weights, initial_value, output_dir=output_dir)
    
    print("Pipeline Complete. Data is ready for Tableau.")

if __name__ == "__main__":
    run_alphapulse()