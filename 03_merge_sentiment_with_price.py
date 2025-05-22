import pandas as pd
import os

# Load files

data_dir = "AAPLStockDataset1980-2025"
price_file = os.path.join(data_dir, "aapl_price_features.csv")
news_file = "StockAAPL/apple_news_data.csv"

df_price = pd.read_csv(price_file, parse_dates=["Date"])
df_news = pd.read_csv(news_file, parse_dates=["date"])

# Step 1: Process news data

df_news["date"] = df_news["date"].dt.date
df_news["date"] = pd.to_datetime(df_news["date"]) 

df_sentiment = df_news.groupby("date").agg({
    "sentiment_polarity": "mean",
    "sentiment_pos": "mean",
    "sentiment_neg": "mean",
    "sentiment_neu": "mean",
    "title": "count" 
}).rename(columns={
    "title": "article_count",
    "date": "Date"
}).reset_index()

# Step 2: Merge with price features

df_merged = pd.merge(df_price, df_sentiment, how="left", left_on="Date", right_on="date")

df_merged.drop(columns=["date"], inplace=True)

df_merged[[
    "sentiment_polarity", "sentiment_pos",
    "sentiment_neg", "sentiment_neu", "article_count"
]] = df_merged[[
    "sentiment_polarity", "sentiment_pos",
    "sentiment_neg", "sentiment_neu", "article_count"
]].fillna(0)

output_path = os.path.join(data_dir, "final_dataset.csv")
df_merged.to_csv(output_path, index=False)

print("✅ Sentiment merged with stock price features.")
print("📁 Saved to:", output_path)
print("📊 Final shape:", df_merged.shape)
print("🧠 Columns:", df_merged.columns.tolist())
