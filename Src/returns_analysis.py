import pandas as pd

# load cleaned dataset
df = pd.read_csv("data/cleaned data/clean_data.csv")

# convert date column
df["Date"] = pd.to_datetime(df["Date"])

# sort by date
df = df.sort_values("Date")

# calculate daily returns
df["Returns"] = df["Close"].pct_change()

# remove first row (NaN return)
df = df.dropna()

# save dataset with returns
df.to_csv("data/cleaned data/returns_data.csv", index=False)

print("Returns calculated successfully")