import pandas as pd
import os
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# === Load enhanced dataset ===
data_dir = "AAPLStockDataset1980-2025"
df = pd.read_csv(os.path.join(data_dir, "final_dataset_enhanced.csv"), parse_dates=["Date"])

drop_cols = ["Date", "return_t+1", "Close", "Open", "High", "Low"]
X = df.drop(columns=drop_cols + ["target"])
y = df["target"]

# === Standardization ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Time-based CV ===
tscv = TimeSeriesSplit(n_splits=5)

# === Random Forest Tuning ===
rf = RandomForestClassifier(random_state=42)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(rf, rf_params, cv=tscv, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_scaled, y)

print("\n🌲 Best RF Params:", rf_grid.best_params_)
print("ROC-AUC:", rf_grid.best_score_)

# === XGBoost Tuning ===
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(xgb, xgb_params, cv=tscv, scoring='roc_auc', n_jobs=-1)
xgb_grid.fit(X_scaled, y)

print("\n🚀 Best XGBoost Params:", xgb_grid.best_params_)
print("ROC-AUC:", xgb_grid.best_score_)
