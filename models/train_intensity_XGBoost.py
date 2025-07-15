import pandas as pd
import numpy as np
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
from sklearn.utils.class_weight import compute_class_weight
import os

# ========================
# 📥 1. Load dữ liệu
# ========================
df = pd.read_csv("D:/Pycharm/weather-new/data/Processed/Intensity/storm_intensity_dataset.csv")

# Nếu có cột 'Season', thêm vào đặc trưng
base_features = ['Rain', 'Temp', 'WindSpeed', 'Pressure', 'Humidity', 'CloudCover', 'WindDirection']
extra_features = ['Month', 'Season'] if 'Month' in df.columns and 'Season' in df.columns else []
features = base_features + extra_features

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

# Mã hóa cột Season thành số (0: Winter, 1: Spring, 2: Summer, 3: Autumn)
season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3}
X_train['Season'] = X_train['Season'].map(season_mapping)
X_test['Season'] = X_test['Season'].map(season_mapping)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Giữ lại tên cột cho XGBoost
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

# ========================
# ⚖️ 5. Tính sample_weight
# ========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
weight_dict = {i: w for i, w in enumerate(class_weights)}
sample_weight = np.array([weight_dict[label] for label in y_train])

# ========================
# 🧠 6. Huấn luyện mô hình XGBoost
# ========================
model = XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    max_depth=6,
    n_estimators=150,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    random_state=42
)
model.fit(X_train_scaled, y_train, sample_weight=sample_weight)

# ========================
# 📊 7. Đánh giá mô hình
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
# 🔷 8. Ma trận nhầm lẫn
# ========================
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test_true, y_pred_true, labels=class_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Dự đoán")
