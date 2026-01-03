import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("dataset.csv")

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

# -------------------------
# Encode labels
# -------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_classes = len(le.classes_)
print("Classes:", le.classes_)

# -------------------------
# Scale features (MANDATORY for LSTM)
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Reshape for LSTM
# Shape: (samples, timesteps=num_features, channels=1)
# -------------------------
X_lstm = X_scaled.reshape(
    X_scaled.shape[0],
    X_scaled.shape[1],
    1
)

# -------------------------
# Train / test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_lstm,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -------------------------
# Build LSTM model
# -------------------------
model = Sequential([
    LSTM(32, input_shape=(X_train.shape[1], 1)),
    Dropout(0.4),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Training with early stopping
# -------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------
# Evaluation
# -------------------------
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
macro_precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)
macro_recall = recall_score(
    y_test, y_pred, average="macro"
)
macro_f1 = f1_score(
    y_test, y_pred, average="macro"
)

print(f"Accuracy:        {acc:.3f}")
print(f"Macro Precision: {macro_precision:.3f}")
print(f"Macro Recall:    {macro_recall:.3f}")
print(f"Macro F1-score:  {macro_f1:.3f}")

print("\nClassification report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=le.classes_
))

# -------------------------
# Confusion matrix
# -------------------------
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
plt.title("LSTM – Confusion Matrix")
plt.tight_layout()
plt.show()



# -------------------------
# Training curves
# -------------------------
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("LSTM Training History")
plt.tight_layout()
plt.show()
