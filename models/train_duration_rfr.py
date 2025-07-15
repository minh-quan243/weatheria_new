import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1️⃣ Load dữ liệu
df = pd.read_csv("D:/Pycharm/weather-new/data/Processed/Duration/storm_duration_dataset.csv", parse_dates=["Datetime"])

# 2️⃣ Mã hóa cột 'Season' thành số
season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
df['Season'] = df['Season'].map(season_mapping)


# 3️⃣ Cập nhật đặc trưng và nhãn
# Nếu Season đã có sẵn trong dữ liệu, ta chỉ cần thêm vào danh sách features
features = [
    'Rain', 'Temp', 'WindSpeed', 'Pressure',
    'Humidity', 'CloudCover', 'WindDirection',
    'Month', 'Season'  # Đảm bảo Season đã có trong dữ liệu
]
target = 'duration_in_storm'
X = df[features]
y = df[target]

# 4️⃣ Tiền xử lý
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# 5️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6️⃣ Huấn luyện Random Forest (tối ưu)
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=16,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 7️⃣ Đánh giá
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("🎯 Đánh giá mô hình Random Forest (đã chỉnh):")
print(f"▪️ MAE: {mae:.2f}")
print(f"▪️ MSE: {mse:.2f}")
print(f"▪️ R²: {r2:.2f}")

# 8️⃣ Lưu model và scaler
os.makedirs("model_use", exist_ok=True)
joblib.dump(model, "model_use/duration_rf_model.pkl")
joblib.dump(scaler, "model_use/duration_scaler.pkl")
print("✅ Đã lưu model và scaler vào thư mục model_use/")

# 9️⃣ Trực quan hóa kết quả

# 9.1 Dự đoán vs Thực tế
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Thực tế (duration)")
plt.ylabel("Dự đoán (duration)")
plt.title("Random Forest: Dự đoán vs Thực tế")
plt.tight_layout()
plt.show()

# 9.2 Phân phối lỗi
errors = y_test - y_pred
plt.figure(figsize=(6, 4))
sns.histplot(errors, bins=40, kde=True, color="orange")
plt.title("Phân phối lỗi dự đoán (Random Forest)")
plt.xlabel("Lỗi (giờ)")
plt.tight_layout()
plt.show()

# 9.3 Feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=model.feature_importances_, y=features, color="teal")
plt.title("Độ quan trọng của các đặc trưng (Random Forest)")
plt.tight_layout()
plt.show()
