import pandas as pd  # type: ignore

# read dataset and skip faulty rows
df = pd.read_csv("data/raw data/msft_stock_data_faulty_raw.csv", skiprows=2)

# fix column names
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

print(df.head())

# remove duplicates
df = df.drop_duplicates()

# remove missing prices
df = df.dropna(subset=["Close"])

# remove invalid prices
df = df[df["Close"] > 0]

# save cleaned dataset
df.to_csv("data/cleaned data/clean_data.csv", index=False)

print("Data cleaned successfully")