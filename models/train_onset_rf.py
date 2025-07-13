# 📦 IMPORT THƯ VIỆN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

# 🧠 1. TẢI DỮ LIỆU
df = pd.read_csv("D:/Pycharm/weather-new/data/Processed/Onset/storm_onset_dataset.csv", parse_dates=["Datetime"])

# ✂️ 2. CHỌN ĐẶC TRƯNG & NHÃN
features = [
    'Rain', 'Temp', 'WindSpeed', 'Pressure',
    'Humidity', 'CloudCover', 'WindDirection'
]
target = "storm_onset"

X = df[features]
print(X)
y = df[target]

# ✅ 3. TÁCH DỮ LIỆU TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🌲 4. HUẤN LUYỆN MÔ HÌNH
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# 📊 5. DỰ ĐOÁN & ĐÁNH GIÁ
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("📈 Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("🎯 Precision:", precision_score(y_test, y_pred))
print("🔁 Recall:", recall_score(y_test, y_pred))
print("🏁 F1 Score:", f1_score(y_test, y_pred))
print("📉 ROC AUC:", roc_auc_score(y_test, y_proba))

# 📊 5.1 VẼ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 📊 5.2 VẼ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# 📊 5.3 FEATURE IMPORTANCE từ RF
importances = model.feature_importances_
sorted_idx = importances.argsort()[::-1]
plt.figure(figsize=(6, 4))
sns.barplot(x=importances[sorted_idx], y=[features[i] for i in sorted_idx])
plt.title("Feature Importance (RF)")
plt.xlabel("Gini Importance")
plt.show()

# 💾 6. LƯU MÔ HÌNH
joblib.dump(model, "model_use/onset_model_random_forest.pkl")
print("✅ Đã lưu model tại: model_use/onset_model_random_forest.pkl")
