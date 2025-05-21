from xgboost import plot_importance
import matplotlib.pyplot as plt
import joblib

xgb = joblib.load("xgb_final_model.pkl")
plot_importance(xgb, importance_type='gain', max_num_features=15)
plt.title("Top XGBoost Feature Importances")
plt.show()
