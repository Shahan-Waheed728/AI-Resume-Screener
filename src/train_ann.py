import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("data/final_dataset.csv")
print(f"Dataset loaded: {df.shape}")

# ---------------------------
# LOAD VECTORIZER
# ---------------------------
vectorizer = joblib.load("models/vectorizer.pkl")

# ---------------------------
# BUILD FEATURES
# ---------------------------
X = []
y = []

for i in range(len(df)):
    resume = str(df.iloc[i]["resume_text"])
    jd = str(df.iloc[i]["job_description"])

    tfidf = vectorizer.transform([resume, jd])
    cosine = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    features = np.hstack([tfidf.toarray().flatten(), cosine])

    X.append(features)
    # ✅ FIX: Target is cosine score (0.0 to 1.0) — NOT multiplied by 100
    y.append(cosine)

X = np.array(X)
y = np.array(y)

print(f"Feature matrix: {X.shape}")
print(f"Target range: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")

# ---------------------------
# TRAIN/TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# ANN MODEL
# ✅ FIX: Output layer uses sigmoid — guarantees output between 0 and 1
# ---------------------------
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    # ✅ sigmoid ensures output is always between 0 and 1
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# ---------------------------
# TRAIN
# ---------------------------
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ---------------------------
# EVALUATE
# ---------------------------
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Loss (MSE): {loss:.4f}")
print(f"✅ Test MAE: {mae:.4f}")

# Verify output range
sample_preds = model.predict(X_test[:5])
print(f"\n✅ Sample predictions (should be 0.0 - 1.0):")
for i, pred in enumerate(sample_preds):
    print(f"  Sample {i+1}: {pred[0]:.4f} ({pred[0]*100:.2f}%)")

# ---------------------------
# SAVE
# ---------------------------
model.save("models/model_ann.h5")
print("\n✅ ANN model saved to models/model_ann.h5")