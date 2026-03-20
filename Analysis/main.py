import os
import numpy as np
# 1. CHANGE: Import 'load_multiple_data' instead of 'load_data'
from data_ingestion import load_multiple_data, calculate_log_returns
from quant_engine import calculate_core_metrics, run_monte_carlo, calculate_var
from tableau_pipeline import export_correlation_matrix, export_monte_carlo_bands, export_historical_metrics
from quant_engine import calculate_core_metrics, run_monte_carlo, calculate_var, generate_efficient_frontier
from tableau_pipeline import export_correlation_matrix, export_monte_carlo_bands, export_historical_metrics, export_efficient_frontier

def run_alphapulse():
    print("Initializing AlphaPulse Quantitative Engine...")
    
    # Output directory for exports
    output_dir = 'Output'
    os.makedirs(output_dir, exist_ok=True)

    # 2. CHANGE: Create a list of all your CSV files
    # Replace these placeholder paths with the actual locations of your multiple CSV files
    portfolio_files = [
        'Data/Cleaned Data/Tech_stock_data.csv',
        'Data/Cleaned Data/Healthcare_stock_data.csv',
        'Data/Cleaned Data/Energy_stock_data.csv',
        'Data/Cleaned Data/Crypto_stock_data.csv'
        # Add as many files as you have!
    ]

    # 3. CHANGE: Use the multiple file loader
    prices = load_multiple_data(portfolio_files)
    
    if prices.empty:
        print("Error: No data loaded. Check your file paths.")
        return

    log_returns = calculate_log_returns(prices)
    
    # 2. Quant Math Setup
    num_assets = len(prices.columns)
    print(f"Successfully loaded {num_assets} assets for the portfolio.")
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
    
    # 4. CHANGE: Export results into the Output directory
    export_correlation_matrix(
        log_returns,
        output_dir=output_dir,
        filename='alphapulse_correlation_matrix.csv'
    )
    export_monte_carlo_bands(
        mc_paths,
        output_dir=output_dir,
        filename='alphapulse_mc_bands.csv'
    )
    export_historical_metrics(
        log_returns,
        rolling_vol_30d,
        weights,
        initial_value,
        output_dir=output_dir,
        filename='alphapulse_historical_risk.csv'
    )
    
    print(f"Pipeline Complete. Data is saved in the '{output_dir}' folder and ready for Tableau.")

    #The Markowitz Efficient Frontier simulation
    print("Optimizing Portfolio (Calculating Efficient Frontier)...")
    
    # Run the simulation engine (5,000 portfolios)
    ef_results, ef_weights = generate_efficient_frontier(mean_returns, cov_matrix)
    
    # Export the results for Tableau
    ef_file = os.path.join(output_dir, 'alphapulse_efficient_frontier.csv')
    asset_names = prices.columns.tolist()
    export_efficient_frontier(ef_results, ef_weights, asset_names, filename=ef_file)
    
    print(f"Pipeline Complete. Data is saved in the '{output_dir}' folder and ready for Tableau.")

if __name__ == "__main__":
    run_alphapulse()