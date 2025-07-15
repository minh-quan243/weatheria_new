import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# 1. Tải dữ liệu từ CSV
# =========================
file_path = 'D:/Pycharm/weather-new/data/Processed/Year/istyphoon_dataset.csv'
data = pd.read_csv(file_path)

# =========================
# 2. Tiền xử lý dữ liệu
# =========================
# Chỉ giữ lại các cột year, month, season và current_typhoon
features = ['year', 'month', 'season']
X = pd.get_dummies(data[features], drop_first=True)  # One-hot encoding cho cột season
y = data['is_typhoon']  # Nhãn: có bão hay không

# =========================
# 3. Chia dữ liệu thành tập huấn luyện và kiểm tra
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 4. Tuning tham số mô hình bằng GridSearchCV
# =========================
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb.XGBClassifier(eval_metric='logloss'),
                           param_grid=param_grid,
                           scoring='accuracy',
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
pred_probs = model.predict_proba(X_test)[:, 1]  # Lấy xác suất dự đoán cho class '1'

# =========================
# 6. Tính các chỉ số đánh giá mô hình
# =========================
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
conf_matrix = confusion_matrix(y_test, preds)

# In kết quả
print(f"Độ chính xác (Accuracy): {accuracy:.4f}")
print(f"Độ chính xác dương (Precision): {precision:.4f}")
print(f"Độ phủ (Recall): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# =========================
# 8. Precision-Recall Curve
# =========================
precision_vals, recall_vals, _ = precision_recall_curve(y_test, pred_probs)

plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, color='b', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# =========================
# 9. ROC Curve
# =========================
fpr, tpr, _ = roc_curve(y_test, pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve (ROC Curve)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# =========================
# 10. Biểu đồ tầm quan trọng đặc trưng
# =========================
plt.figure(figsize=(8, 6))
xgb.plot_importance(model, importance_type='weight', max_num_features=10, title="Tầm quan trọng đặc trưng")
plt.tight_layout()
plt.show()

# =========================
# 11. Lưu mô hình
# =========================
model_filename = 'model_use/istyphoon_model.pkl'
joblib.dump(model, model_filename)
print(f"✅ Mô hình đã được lưu tại: {model_filename}")
