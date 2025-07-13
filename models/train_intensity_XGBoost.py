import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

# ========================
# 📥 1. Load dữ liệu
# ========================
df = pd.read_csv("D:/Pycharm/weather-new/data/Processed/Intensity/storm_intensity1_dataset.csv")

features = ['Rain', 'Temp', 'WindSpeed', 'Pressure', 'Humidity', 'CloudCover', 'WindDirection']
target = 'storm_category'

X = df[features]
y = df[target]

# ========================
# 🔤 2. Encode nhãn phân loại
# ========================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_labels = le.classes_
num_classes = len(class_labels)

# ========================
# ✅ 3. Tách dữ liệu train/test
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
)

# ========================
# 🧼 4. Chuẩn hóa dữ liệu
# ========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# 🧠 5. Huấn luyện mô hình XGBoost
# ========================
model = XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ========================
# 📊 6. Đánh giá mô hình
# ========================
y_pred = model.predict(X_test_scaled)
y_test_true = le.inverse_transform(y_test)
y_pred_true = le.inverse_transform(y_pred)

print("📈 Classification Report:\n", classification_report(y_test_true, y_pred_true))
print("✅ Accuracy:", accuracy_score(y_test_true, y_pred_true))
print("🎯 Precision (macro):", precision_score(y_test_true, y_pred_true, average="macro"))
print("🔁 Recall (macro):", recall_score(y_test_true, y_pred_true, average="macro"))
print("🏁 F1 Score (macro):", f1_score(y_test_true, y_pred_true, average="macro"))

# 🎯 ROC-AUC với multi-class (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=range(num_classes))
y_proba = model.predict_proba(X_test_scaled)
auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
print("📉 ROC-AUC (macro):", auc)

# ========================
# 🔷 7. Ma trận nhầm lẫn
# ========================
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test_true, y_pred_true, labels=class_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.show()

# ========================
# 🌟 8. Feature Importance
# ========================
plt.figure(figsize=(8, 6))
plot_importance(model, importance_type='gain', show_values=False)
plt.title("Feature Importance (Gain) - XGBoost")
plt.tight_layout()
plt.show()

# ========================
# 💾 9. Lưu mô hình, encoder và scaler
# ========================
joblib.dump(model, "model_use/storm_intensity_xgboost.pkl")
joblib.dump(le, "model_use/label_encoder_intensity.pkl")
joblib.dump(scaler, "model_use/scaler_intensity.pkl")

print("✅ Đã lưu mô hình, label encoder và scaler vào thư mục model_use/")
