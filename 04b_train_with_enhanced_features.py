import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

data_dir = "AAPLStockDataset1980-2025"
df = pd.read_csv(os.path.join(data_dir, "final_dataset_enhanced.csv"), parse_dates=["Date"])

drop_cols = ["Date", "return_t+1", "Close", "Open", "High", "Low"]
X = df.drop(columns=drop_cols + ["target"])
y = df["target"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_preds = log_reg.predict(X_test_scaled)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

xgb = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=200, subsample=0.8, eval_metric="logloss", random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 Results for {name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
    print("ROC-AUC:", roc_auc_score(y_true, y_pred))

evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("XGBoost", y_test, xgb_preds)
