import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("data/final_dataset.csv")
print(f"✅ Dataset loaded: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

# ---------------------------
# LOAD VECTORIZER
# ---------------------------
vectorizer = joblib.load("models/vectorizer.pkl")

# ---------------------------
# BUILD FEATURES
# ---------------------------
print("Building features...")
X, y = [], []

for i in range(len(df)):
    resume = str(df.iloc[i]["resume_text"])
    jd     = str(df.iloc[i]["job_description"])

    tfidf  = vectorizer.transform([resume, jd])
    cosine = cosine_similarity(tfidf[0], tfidf[1])[0][0]

    features = np.hstack([tfidf.toarray().flatten(), cosine])
    X.append(features)
    # ✅ Label is 0 or 1 — ANN learns classification
    y.append(df.iloc[i]["label"])

    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{len(df)} rows...")

X = np.array(X)
y = np.array(y, dtype=np.float32)
print(f"\n✅ Feature matrix: {X.shape}")

# ---------------------------
# TRAIN / TEST SPLIT (70/15/15)
# ---------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nTrain: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ---------------------------
# CLASS WEIGHTS (handle imbalance)
# ---------------------------
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
total = len(y_train)
weight_0 = (1 / neg) * (total / 2.0)
weight_1 = (1 / pos) * (total / 2.0)
class_weights = {0: weight_0, 1: weight_1}
print(f"\nClass weights: {class_weights}")

# ---------------------------
# ANN MODEL
# ✅ sigmoid output → score between 0 and 1
# ✅ binary_crossentropy → correct loss for classification
# ---------------------------
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    # ✅ sigmoid → output always between 0 and 1
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

model.summary()

# ---------------------------
# CALLBACKS
# ---------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, verbose=1
    )
]

# ---------------------------
# TRAIN
# ---------------------------
print("\nTraining ANN...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ---------------------------
# EVALUATION
# ---------------------------
print("\n── Test Evaluation ──")
loss, acc, prec, rec = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {2*(prec*rec)/(prec+rec):.4f}")

# Predictions
y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

print(f"\n── Classification Report ──")
print(classification_report(y_test, y_pred,
      target_names=['Not Qualified', 'Qualified']))

print(f"\n── Confusion Matrix ──")
print(confusion_matrix(y_test, y_pred))

# Verify output range
print(f"\n── Sample Predictions ──")
for i in range(5):
    prob = y_pred_prob[i]
    print(f"  Sample {i+1}: {prob:.4f} ({prob*100:.1f}%) → "
          f"{'Qualified' if prob >= 0.5 else 'Not Qualified'}")

# ---------------------------
# SAVE
# ---------------------------
os.makedirs("models", exist_ok=True)
model.save("models/model_ann.h5")
print("\n✅ ANN model saved to models/model_ann.h5")