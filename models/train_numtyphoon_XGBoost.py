import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# 1. Tải dữ liệu từ CSV
# =========================
file_path = 'D:/Pycharm/weather-new/data/Processed/Year/numtyphoon_dataset.csv'
data = pd.read_csv(file_path)

# =========================
# 2. Tiền xử lý dữ liệu
# =========================
# Chỉ giữ lại các cột year, month, season và số bão trong tháng hiện tại
features = ['year', 'month', 'season']
X = pd.get_dummies(data[features], drop_first=True)  # One-hot encoding cho cột season
y = data['num_typhoons']  # Nhãn: số bão trong tháng hiện tại

# =========================
# 3. Chia dữ liệu thành tập huấn luyện và kiểm tra
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 4. Hyperparameter Tuning với GridSearchCV
# =========================
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'alpha': [0, 0.5, 1],
    'lambda': [0, 0.5, 1]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(eval_metric='rmse'),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3,
                           verbose=1,
                           n_jobs=-1)

# Tuning tham số
grid_search.fit(X_train, y_train)

# In kết quả tìm được tham số tốt nhất
print(f"Best parameters found: {grid_search.best_params_}")

# Lấy mô hình đã được tối ưu hóa
model = grid_search.best_estimator_

# =========================
# 5. Dự đoán trên tập kiểm tra
# =========================
preds = model.predict(X_test)

# =========================
# 6. Tính các chỉ số đánh giá mô hình
# =========================
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# In kết quả
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2): {r2:.4f}")

# =========================
# 7. Biểu đồ dự đoán so với giá trị thực tế
# =========================
plt.figure(figsize=(6, 5))
plt.scatter(y_test, preds)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title('Dự đoán so với giá trị thực tế')
plt.tight_layout()
plt.show()

# =========================
# 8. Biểu đồ tầm quan trọng đặc trưng
# =========================
plt.figure(figsize=(8, 6))
xgb.plot_importance(model, importance_type='weight', max_num_features=10, title="Tầm quan trọng đặc trưng")
plt.tight_layout()
plt.show()

# =========================
# 9. Lưu mô hình
# =========================
model_filename = 'model_use/numtyphoon_model.pkl'
joblib.dump(model, model_filename)
print(f"✅ Mô hình đã được lưu tại: {model_filename}")
