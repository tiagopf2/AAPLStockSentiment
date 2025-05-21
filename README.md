# 📈 AAPL Stock Sentiment Classifier

This project uses financial news sentiment and technical indicators to predict whether Apple Inc. (AAPL) stock will go up or down the next day.

🧠 Built with real-world news and stock price data from 2010–2025, it combines Natural Language Processing, machine learning, and market features to train a binary classification model.

---

## 🚀 Objective

Predict whether Apple's stock price will increase the next day (binary classification: up = 1, down = 0), using:

- Daily stock price and volume data
- Sentiment scores from financial news articles
- Technical indicators like RSI and MACD
- Temporal and calendar patterns (day of week, month end)

---

## 🔧 How It Works

1. **Data Sources**
   - `aapl_us_d.csv` and `aapl_us_2025.csv`: AAPL daily price and volume data (2010–2025)
   - `apple_news_data.csv`: News articles with sentiment scores for AAPL

2. **Feature Engineering**
   - Lagged returns, volatility, rolling averages
   - News sentiment aggregates (mean polarity, positive/negative proportions)
   - Calendar features (weekday, month-end)
   - RSI, MACD (technical indicators)

3. **Models Trained**
   - Logistic Regression (baseline)
   - Random Forest (tuned)
   - XGBoost (tuned — best performer, ROC-AUC ≈ 0.60)

4. **Evaluation**
   - Accuracy, F1-score, ROC-AUC
   - Time-based train/test split to avoid lookahead bias
   - Feature importance visualization

---

## 📊 Results

| Model           | Accuracy | ROC-AUC |
|----------------|----------|---------|
| Logistic Reg.  | ~55.2%   | ~0.537  |
| Random Forest  | 57.9%    | 0.583   |
| XGBoost        | 57.8%    | **0.581** ✅

💡 These results show that sentiment and technical signals can improve predictive performance compared to price alone — even if modestly, which is common in finance.

---

## 📁 Project Structure

