import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (CSV already has headers)
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

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(dict(zip(le.classes_, le.transform(le.classes_))))

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

# Evaluation
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
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
plt.title("Random Forest – Confusion Matrix")
plt.tight_layout()
plt.show()

report = classification_report(
    y_test,
    y_pred,
    target_names=le.classes_,
    output_dict=True
)

recall_df = pd.DataFrame(report).T[["recall", "support"]]
print(recall_df)

importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importances.head(15))

plt.figure(figsize=(8, 6))
importances.head(12).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 12 Feature Importances – Random Forest")
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

rf_reduced = RandomForestClassifier(
    n_estimators=400,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

rf_reduced.fit(X_train_r, y_train_r)

y_pred_r = rf_reduced.predict(X_test_r)

print("Reduced-feature Accuracy:", accuracy_score(y_test_r, y_pred_r))
print(classification_report(y_test_r, y_pred_r, target_names=le.classes_))

