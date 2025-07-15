import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# 1. Tải dữ liệu từ CSV
# =========================
file_path = 'D:/Pycharm/weather-new/data/Processed/Year/region_dataset.csv'
data = pd.read_csv(file_path)

# =========================
# 2. Tiền xử lý dữ liệu
# =========================
# Mã hóa thủ công các vùng bão thành các số nguyên
region_mapping = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Unknown': 4, 'Central': 5}
data['region'] = data['region'].map(region_mapping)  # Áp dụng mã hóa thủ công

# Kiểm tra và loại bỏ dòng có giá trị NaN trong 'region'
if data['region'].isnull().sum() > 0:
    print(f"Đã tìm thấy {data['region'].isnull().sum()} giá trị NaN trong cột 'region'.")
    data = data.dropna(subset=['region'])  # Loại bỏ dòng có NaN trong cột 'region'

# Chỉ giữ lại các cột year, month, season và region (vùng bão)
features = ['year', 'month', 'season']
X = pd.get_dummies(data[features], drop_first=True)  # One-hot encoding cho cột season

# Nhãn là vùng bão xuất hiện (North, South, East, West hoặc Unknown) đã mã hóa thành số nguyên
y = data['region']  # Nhãn: vùng bão đã mã hóa

# =========================
# 3. Chia dữ liệu thành tập huấn luyện và kiểm tra
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# 4. Huấn luyện mô hình XGBoost (Phân loại)
# =========================
model = xgb.XGBClassifier(eval_metric='mlogloss')  # Đã xóa tham số 'use_label_encoder'

# Chuyển sang mô hình One-vs-Rest (OvR) để làm việc với đa lớp
ovr_model = OneVsRestClassifier(model)
ovr_model.fit(X_train, y_train)

# =========================
# 5. Dự đoán trên tập kiểm tra
# =========================
preds = ovr_model.predict(X_test)
pred_probs = ovr_model.predict_proba(X_test)

# =========================
# 6. Tính các chỉ số đánh giá mô hình
# =========================
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average='weighted', zero_division=1)
recall = recall_score(y_test, preds, average='weighted', zero_division=1)
f1 = f1_score(y_test, preds, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_test, preds)

# In kết quả
print(f"Độ chính xác (Accuracy): {accuracy:.4f}")
print(f"Độ chính xác dương (Precision): {precision:.4f}")
print(f"Độ phủ (Recall): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# =========================
# 7. Log Loss (Logarithmic Loss)
# =========================
# Lấy các lớp trong y_true để tránh sự không khớp với y_prob
labels = list(set(y))  # Lấy các lớp duy nhất có trong y_true
print(f"Log Loss (Logarithmic Loss): {log_loss(y_test, pred_probs, labels=labels):.4f}")

# =========================
# 9. Biểu đồ Precision-Recall cho mỗi lớp
# =========================
from sklearn.metrics import precision_recall_curve

# Lấy số lớp thực tế có trong mô hình
# Lấy số lớp từ y_test
n_classes = len(set(y_test))  # Số lớp trong y_test

for i in range(n_classes):
    if i in y_test.values:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test == i, ovr_model.predict_proba(X_test)[:, i])
        # Đảm bảo mỗi lớp có label khi vẽ đồ thị
        plt.plot(recall_vals, precision_vals, label=label_encoder.classes_[i] if 'label_encoder' in locals() else f'Class {i}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend(loc='best')  # Bây giờ plt.legend() sẽ hiển thị vì mỗi lớp có nhãn
plt.tight_layout()
plt.show()

# =========================
# 10. Biểu đồ tầm quan trọng đặc trưng
# =========================
plt.figure(figsize=(8, 6))
xgb.plot_importance(ovr_model.estimators_[0], importance_type='weight', max_num_features=10, title="Tầm quan trọng đặc trưng")
plt.tight_layout()
plt.show()

# =========================
# 11. Lưu mô hình
# =========================
model_filename = 'model_use/region_model.pkl'
joblib.dump(ovr_model, model_filename)
print(f"✅ Mô hình đã được lưu tại: {model_filename}")
