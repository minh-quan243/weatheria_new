# 📦 IMPORT THƯ VIỆN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from xgboost import XGBClassifier, plot_importance
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
y = df[target]

# ✅ 3. TÁCH DỮ LIỆU TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ⚖️ 4. TÍNH scale_pos_weight
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 🔁 5. HUẤN LUYỆN MÔ HÌNH XGBoost
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_weight,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 📊 6. DỰ ĐOÁN & ĐÁNH GIÁ
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("📈 Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("🎯 Precision:", precision_score(y_test, y_pred))
print("🔁 Recall:", recall_score(y_test, y_pred))
print("🏁 F1 Score:", f1_score(y_test, y_pred))
print("📉 ROC AUC:", roc_auc_score(y_test, y_proba))

# 🔍 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.show()

# 📈 8. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🔥 9. Feature Importance
plt.figure(figsize=(6, 4))
plot_importance(model, importance_type="gain", show_values=False)
plt.title("Feature Importance (Gain) - XGBoost")
plt.tight_layout()
plt.show()

# 💾 10. LƯU MÔ HÌNH
joblib.dump(model, "model_use/onset_model_xgboost.pkl")
print("✅ Đã lưu mô hình vào: model_use/onset_model_xgboost.pkl")
