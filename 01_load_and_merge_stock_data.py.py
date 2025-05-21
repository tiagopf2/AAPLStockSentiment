import pandas as pd

df_2025 = pd.read_csv("AAPLStockDataset1980-2025/aapl_us_2025.csv", parse_dates=["Date"])
df_hist = pd.read_csv("AAPLStockDataset1980-2025/aapl_us_d.csv", parse_dates=["Date"])

df_price = pd.concat([df_hist, df_2025], ignore_index=True).sort_values("Date")
