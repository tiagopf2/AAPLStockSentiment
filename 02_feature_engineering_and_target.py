import pandas as pd
import os

# Load and merge stock data

data_dir = "AAPLStockDataset1980-2025"
file_2025 = os.path.join(data_dir, "aapl_us_2025.csv")
file_hist = os.path.join(data_dir, "aapl_us_d.csv")

df_2025 = pd.read_csv(file_2025, parse_dates=["Date"])
df_hist = pd.read_csv(file_hist, parse_dates=["Date"])

df_price = pd.concat([df_hist, df_2025], ignore_index=True)
df_price = df_price.sort_values("Date").reset_index(drop=True)

# Step 1: Create Return and Target

df_price["return_t"] = (df_price["Close"] - df_price["Open"]) / df_price["Open"]

df_price["return_t+1"] = df_price["Close"].shift(-1) / df_price["Close"] - 1

df_price["target"] = (df_price["return_t+1"] > 0).astype(int)

# Step 2: Create Features

df_price["volatility"] = (df_price["High"] - df_price["Low"]) / df_price["Open"]

df_price["return_t-1"] = df_price["return_t"].shift(1)

df_price["ma_3"] = df_price["Close"].rolling(window=3).mean()
df_price["ma_5"] = df_price["Close"].rolling(window=5).mean()

df_price["ma_volume_5"] = df_price["Volume"].rolling(window=5).mean()

df_price["momentum_3"] = df_price["Close"] - df_price["Close"].shift(3)

# Step 3: Clean Dataset

df_price = df_price.dropna().reset_index(drop=True)

output_path = os.path.join(data_dir, "aapl_price_features.csv")
df_price.to_csv(output_path, index=False)

print("✅ Feature engineering complete.")
print("📁 Saved to:", output_path)
print("📊 Final shape:", df_price.shape)
print("🧠 Columns:", df_price.columns.tolist())
