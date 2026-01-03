import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# Load dataset (CSV already has headers)
df = pd.read_csv("dataset.csv")

print(df.head())
print(df.shape)

# Target
y = df["leak_location"]

# Features
X = df.drop(columns=[
    "case",
    "leak_binary",
    "leak_location"
])

print(X.shape)
print(y.value_counts())


le = LabelEncoder()
y_encoded = le.fit_transform(y)

class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Class mapping:", class_mapping)

num_classes = len(le.classes_)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,      # L1 regularization
    reg_lambda=1.0,     # L2 regularization
    eval_metric="mlogloss",
    random_state=42
)


xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("XGBoost – Confusion Matrix")
plt.tight_layout()
plt.show()

importances = pd.Series(
    xgb.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importances.head(15))


plt.figure(figsize=(8, 6))
importances.head(12).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 12 Feature Importances – XGBoost")
plt.xlabel("Importance (Gain)")
plt.tight_layout()
plt.show()


important_features = importances[importances > 0.01].index
X_reduced = X[important_features]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reduced,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

xgb_reduced = XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=42
)

xgb_reduced.fit(X_train_r, y_train_r)

y_pred_r = xgb_reduced.predict(X_test_r)

print("Reduced-feature Accuracy:", accuracy_score(y_test_r, y_pred_r))
print(classification_report(y_test_r, y_pred_r, target_names=le.classes_))
