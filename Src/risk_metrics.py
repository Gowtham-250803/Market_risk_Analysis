import pandas as pd  # type: ignore
import numpy as np  # type: ignore

# load dataset with returns
df = pd.read_csv("data/cleaned data/returns_data.csv")

# calculate volatility
volatility = df["Returns"].std()

# calculate Value at Risk (95%)
VaR_95 = np.percentile(df["Returns"], 5)

print("Volatility:", volatility)
print("Value at Risk (95%):", VaR_95)