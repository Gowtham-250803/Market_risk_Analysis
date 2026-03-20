import numpy as np
import pandas as pd

def calculate_core_metrics(log_returns):
    """Returns mean daily returns, covariance matrix, and 30-day rolling volatility."""
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    rolling_vol_30d = log_returns.rolling(window=30).std() * np.sqrt(252)
    
    return mean_returns, cov_matrix, rolling_vol_30d

def run_monte_carlo(mean_returns, cov_matrix, weights, initial_value=1000000, num_simulations=10000, time_horizon=252):
    """Executes the Monte Carlo simulation using multivariate normal distributions."""
    # Convert inputs to numpy arrays
    if hasattr(mean_returns, 'values'):
        mean = mean_returns.values
    else:
        mean = np.asarray(mean_returns)

    cov = np.asarray(cov_matrix)
    weights = np.asarray(weights)

    # Remove any assets with NaNs in mean or covariance
    nan_mean_mask = np.isnan(mean)
    nan_cov_mask = np.isnan(cov).any(axis=0)
    valid_mask = ~(nan_mean_mask | nan_cov_mask)

    if not valid_mask.all():
        mean = mean[valid_mask]
        cov = cov[np.ix_(valid_mask, valid_mask)]
        weights = weights[valid_mask]

    # Ensure covariance is symmetric
    cov = (cov + cov.T) / 2.0

    # Add tiny jitter to diagonal to improve numerical stability
    eps = 1e-8
    cov = cov + np.eye(cov.shape[0]) * eps

    # If covariance is not positive semidefinite, project to nearest PSD via eigen clipping
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals[eigvals < 0] = 0.0
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        cov = (cov + cov.T) / 2.0
        cov = cov + np.eye(cov.shape[0]) * eps

    # Normalize weights to sum to 1 (if not already)
    if weights.sum() != 0:
        weights = weights / weights.sum()

    # Simulate the daily returns for all paths
    simulated_daily_returns = np.random.multivariate_normal(
        mean, cov, (num_simulations, time_horizon)
    )

    # Calculate portfolio return for each day across all simulations
    simulated_port_returns = np.dot(simulated_daily_returns, weights)

    # Calculate cumulative returns and absolute portfolio values
    cumulative_returns = np.exp(np.cumsum(simulated_port_returns, axis=1))
    portfolio_paths = initial_value * cumulative_returns

    return portfolio_paths

def calculate_var(portfolio_paths, initial_value=1000000, confidence_level=5):
    """Calculates the Value at Risk based on the simulated end values."""
    simulated_end_values = portfolio_paths[:, -1] # Get the final day's values
    percentile_worst_case = np.percentile(simulated_end_values, confidence_level)
    var_value = initial_value - percentile_worst_case
    return var_value

def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=0.02):
    """
    Simulates thousands of random portfolio weights to find the Efficient Frontier.
    Returns an array of Returns, Volatilities, Sharpe Ratios, and the corresponding Weights.
    """
    num_assets = len(mean_returns)
    
    # Arrays to hold our simulation results
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        # Generate random weights and normalize them so they sum to 1.0 (100%)
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        # Calculate Expected Annualized Return
        # Assuming 252 trading days in a year
        portfolio_return = np.sum(mean_returns * weights) * 252
        
        # Calculate Expected Annualized Volatility (Risk)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
        
        # Store results
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        results[2,i] = sharpe_ratio
        
    return results, weights_record