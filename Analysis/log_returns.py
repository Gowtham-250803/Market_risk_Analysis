import numpy as np # type: ignore
import pandas as pd  # type: ignore
import sys

# Load price data (adjust the path/filename to your actual data source)
prices_df = pd.read_csv(
    r"data/cleaned data/returns_data.csv",
    index_col=0,
    parse_dates=True,
)

# Basic cleaning: replace infinities and drop empty columns
prices_df = prices_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

# Drop assets that contain non-positive prices (log undefined for <= 0)
non_positive_cols = (prices_df <= 0).any(axis=0)
if non_positive_cols.any():
    dropped = non_positive_cols.sum()
    prices_df = prices_df.loc[:, ~non_positive_cols]
    print(f"Dropped {dropped} asset(s) with non-positive prices.")

if prices_df.shape[1] == 0:
    print("No valid asset columns remain after cleaning. Exiting.")
    sys.exit(1)

# Calculate daily log returns
log_returns = np.log(prices_df / prices_df.shift(1)).dropna()

# Calculate mean daily returns and the covariance matrix
mean_returns = log_returns.mean()
cov_matrix = log_returns.cov()

# Ensure covariance matrix has no NaNs and is symmetric
if cov_matrix.isnull().any().any():
    print("Covariance matrix contains NaNs; filling with zeros where necessary.")
    cov_matrix = cov_matrix.fillna(0.0)
cov_matrix = (cov_matrix + cov_matrix.T) / 2

def _make_pos_def(mat, eps=1e-8):
    # Force positive (semi-)definite by eigenvalue clipping
    vals, vecs = np.linalg.eigh(mat)
    if np.any(vals <= 0):
        vals_clipped = np.clip(vals, eps, None)
        return (vecs @ np.diag(vals_clipped) @ vecs.T)
    return mat

# Try Cholesky; if that fails, regularize the matrix
cov_values = cov_matrix.values
try:
    np.linalg.cholesky(cov_values)
except np.linalg.LinAlgError:
    cov_values = _make_pos_def(cov_values, eps=1e-8)
    cov_matrix = pd.DataFrame(cov_values, index=mean_returns.index, columns=mean_returns.index)
    print("Regularized covariance matrix to be positive-definite.")

# Historical Volatility (Annualized)
historical_volatility = log_returns.std() * np.sqrt(252)

# 30-Day Rolling Volatility (Annualized)
rolling_volatility_30d = log_returns.rolling(window=30).std() * np.sqrt(252)

# Define portfolio parameters
num_assets = len(prices_df.columns)
weights = np.full(num_assets, 1/num_assets) # Assuming equal weights for this example
initial_portfolio_value = 1000000 # Example: $1M baseline

# Calculate baseline Portfolio Variance using NumPy dot products
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

# --- MONTE CARLO SETUP ---
num_simulations = 10000
time_horizon = 252 # Simulating 1 year out

# Initialize array to hold the final portfolio values for all 10k simulations
simulated_end_values = np.zeros(num_simulations)

# We use Cholesky decomposition to generate correlated random variables
# Alternatively, np.random.multivariate_normal handles this under the hood:
for i in range(num_simulations):
    # Simulate correlated daily returns for the given time horizon
    try:
        sim_returns = np.random.multivariate_normal(mean_returns, cov_values, time_horizon)
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fallback: use independent normals with variances from diagonal
        print("multivariate_normal failed during simulation; falling back to independent normals.")
        sim_returns = np.random.normal(loc=mean_returns.values, scale=np.sqrt(np.diag(cov_values)), size=(time_horizon, len(mean_returns)))
    
    # Calculate the simulated portfolio return for each day
    sim_portfolio_returns = np.dot(sim_returns, weights)
    
    # Calculate the cumulative return and apply it to the initial value
    cumulative_return = np.exp(np.sum(sim_portfolio_returns))
    simulated_end_values[i] = initial_portfolio_value * cumulative_return

# Calculate the 5th percentile of the simulated end values
percentile_5th = np.percentile(simulated_end_values, 5)

# VaR is the difference between the initial value and the 5th percentile worst-case scenario
var_95 = initial_portfolio_value - percentile_5th

print(f"95% VaR: ${var_95:,.2f}")
print(f"This means there is a 5% chance the portfolio will lose more than ${var_95:,.2f} over the next year.")

# Calculate Historical Portfolio Daily Returns
historical_portfolio_returns = np.dot(log_returns, weights)

# Calculate Historical VaR (1-day, 95% confidence)
historical_1d_var_percent = np.percentile(historical_portfolio_returns, 5)
historical_1d_var_dollar = initial_portfolio_value * (1 - np.exp(historical_1d_var_percent))

# Calculate Simulated 1-day VaR to compare
try:
    simulated_1d_returns = np.random.multivariate_normal(mean_returns, cov_values, num_simulations)
except (np.linalg.LinAlgError, ValueError):
    print("multivariate_normal failed for 1-day simulation; using independent normals.")
    simulated_1d_returns = np.random.normal(loc=mean_returns.values, scale=np.sqrt(np.diag(cov_values)), size=(num_simulations, len(mean_returns)))
simulated_1d_port_returns = np.dot(simulated_1d_returns, weights)
simulated_1d_var_percent = np.percentile(simulated_1d_port_returns, 5)

print(f"Historical 1-Day 95% Return Drop: {historical_1d_var_percent:.4%}")
print(f"Simulated 1-Day 95% Return Drop:  {simulated_1d_var_percent:.4%}")

