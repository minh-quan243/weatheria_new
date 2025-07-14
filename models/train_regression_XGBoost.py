import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import os

# ✅ 1. Load dữ liệu
df = pd.read_csv("D:/Pycharm/weather-new/data/Processed/Regression/storm_regression_dataset.csv")

# ✅ 2. Giữ lại những dòng có max_wind > 0 (điểm có bão)
df = df[df["max_wind"] > 0]

# ✅ 3. Đặc trưng và nhãn
features = ['Rain', 'Temp', 'WindSpeed', 'Pressure', 'Humidity', 'CloudCover', 'WindDirection']
target = "max_wind"
X = df[features]
y = df[target]

# ✅ 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ✅ 6. Gán sample_weight cho điểm gió lớn
# Điểm nào gió mạnh hơn 100 km/h thì nhân trọng số 5, còn lại là 1
weights = np.where(y_train > 100, 5, 1)

# ✅ 7. Huấn luyện XGBoost Regressor với tham số mạnh hơn
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.5,
    reg_lambda=1,
    random_state=42
)
model.fit(X_train, y_train, sample_weight=weights)

# ✅ 8. Đánh giá
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Đánh giá mô hình XGBoost:")
print(f"▪MAE: {mae:.2f}")
print(f"▪MSE: {mse:.2f}")
print(f"▪R²: {r2:.2f}")

# ✅ 9. Lưu model & scaler
os.makedirs("model_use", exist_ok=True)
joblib.dump(model, "model_use/regression_xgboost_maxwind.pkl")
joblib.dump(scaler, "model_use/regression_scaler.pkl")
print("✅ Đã lưu model và scaler.")

# ✅ 10. Trực quan hóa
# 10.1 Thực tế vs Dự đoán
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Thực tế (max_wind)")
plt.ylabel("Dự đoán (max_wind)")
plt.title("XGBoost: Dự đoán vs Thực tế")
plt.tight_layout()
plt.show()

# 10.2 Phân phối lỗi
errors = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(errors, bins=50, kde=True, color="tomato")
plt.title("Phân phối lỗi dự đoán (XGBoost)")
plt.xlabel("Lỗi")
plt.tight_layout()
plt.show()

# 10.3 Feature Importance
plt.figure(figsize=(8, 5))
sns.barplot(x=model.feature_importances_, y=features, color="teal")
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()
