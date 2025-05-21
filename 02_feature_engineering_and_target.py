import pandas as pd
import os

# === Step 0: Load and merge stock data ===

# Define file paths
data_dir = "AAPLStockDataset1980-2025"
file_2025 = os.path.join(data_dir, "aapl_us_2025.csv")
file_hist = os.path.join(data_dir, "aapl_us_d.csv")

# Load data
df_2025 = pd.read_csv(file_2025, parse_dates=["Date"])
df_hist = pd.read_csv(file_hist, parse_dates=["Date"])

# Merge and sort by date
df_price = pd.concat([df_hist, df_2025], ignore_index=True)
df_price = df_price.sort_values("Date").reset_index(drop=True)

# === Step 1: Create Return and Target ===

# Return for current day
df_price["return_t"] = (df_price["Close"] - df_price["Open"]) / df_price["Open"]

# Return for next day (used to build the target)
df_price["return_t+1"] = df_price["Close"].shift(-1) / df_price["Close"] - 1

# Binary target: 1 if next day's return is positive, 0 otherwise
df_price["target"] = (df_price["return_t+1"] > 0).astype(int)

# === Step 2: Create Features ===

# Volatility feature
df_price["volatility"] = (df_price["High"] - df_price["Low"]) / df_price["Open"]

# Lagged return
df_price["return_t-1"] = df_price["return_t"].shift(1)

# 3-day and 5-day moving averages of Close price
df_price["ma_3"] = df_price["Close"].rolling(window=3).mean()
df_price["ma_5"] = df_price["Close"].rolling(window=5).mean()

# 5-day moving average of Volume
df_price["ma_volume_5"] = df_price["Volume"].rolling(window=5).mean()

# Momentum feature: price change from 3 days ago
df_price["momentum_3"] = df_price["Close"] - df_price["Close"].shift(3)

# === Step 3: Clean Dataset ===

# Drop rows with NaNs caused by rolling and shifting
df_price = df_price.dropna().reset_index(drop=True)

# Optional: Save processed data to file
output_path = os.path.join(data_dir, "aapl_price_features.csv")
df_price.to_csv(output_path, index=False)

print("✅ Feature engineering complete.")
print("📁 Saved to:", output_path)
print("📊 Final shape:", df_price.shape)
print("🧠 Columns:", df_price.columns.tolist())
