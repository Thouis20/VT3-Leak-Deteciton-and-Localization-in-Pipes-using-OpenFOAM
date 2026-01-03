import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

from xgboost import XGBClassifier

df = pd.read_csv("dataset.csv")

y = df["leak_location"]
X = df.drop(columns=["case", "leak_binary", "leak_location"])

le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_classes = len(le.classes_)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

acc_scores = []
macro_recall_scores = []
macro_precision_scores = []
macro_f1_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    model = XGBClassifier(
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
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_recall = recall_score(y_test, y_pred, average="macro")
    macro_precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    acc_scores.append(acc)
    macro_recall_scores.append(macro_recall)
    macro_precision_scores.append(macro_precision)
    macro_f1_scores.append(macro_f1)

    print(
        f"Fold {fold}: "
        f"Accuracy={acc:.3f}, "
        f"Macro Precision={macro_precision:.3f}, "
        f"Macro Recall={macro_recall:.3f}, "
        f"Macro F1={macro_f1:.3f}"
    )

print("\nXGBoost 5-Fold Cross-Validation Results")
print(f"Accuracy:        {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}")
print(f"Macro Precision: {np.mean(macro_precision_scores):.3f} ± {np.std(macro_precision_scores):.3f}")
print(f"Macro Recall:    {np.mean(macro_recall_scores):.3f} ± {np.std(macro_recall_scores):.3f}")
print(f"Macro F1-score:  {np.mean(macro_f1_scores):.3f} ± {np.std(macro_f1_scores):.3f}")
