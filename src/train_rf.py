import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)
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

    # TF-IDF vectors + cosine score as extra feature
    features = np.hstack([tfidf.toarray().flatten(), cosine])
    X.append(features)
    y.append(df.iloc[i]["label"])

    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{len(df)} rows...")

X = np.array(X)
y = np.array(y)
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
# RANDOM FOREST MODEL
# ---------------------------
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # handles imbalance (2484 vs 1017)
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ---------------------------
# CROSS VALIDATION
# ---------------------------
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
print(f"\n✅ 5-Fold CV F1 Scores: {cv_scores}")
print(f"✅ Mean CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---------------------------
# EVALUATION
# ---------------------------
y_pred_val  = rf.predict(X_val)
y_pred_test = rf.predict(X_test)

print(f"\n── Validation Results ──")
print(f"Accuracy  : {accuracy_score(y_val, y_pred_val):.4f}")
print(f"Precision : {precision_score(y_val, y_pred_val):.4f}")
print(f"Recall    : {recall_score(y_val, y_pred_val):.4f}")
print(f"F1 Score  : {f1_score(y_val, y_pred_val):.4f}")

print(f"\n── Test Results ──")
print(f"Accuracy  : {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_test):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_test):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred_test):.4f}")

print(f"\n── Classification Report ──")
print(classification_report(y_test, y_pred_test,
      target_names=['Not Qualified', 'Qualified']))

print(f"\n── Confusion Matrix ──")
print(confusion_matrix(y_test, y_pred_test))

# ---------------------------
# SAVE MODEL
# ---------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/model_rf.pkl")
print("\n✅ Random Forest saved to models/model_rf.pkl")