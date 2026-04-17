import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# ---------------------------
# LOAD DATA (FIXED)
# ---------------------------
df = pd.read_csv("data/final_dataset.csv")

# ---------------------------
# CREATE COMBINED TEXT FOR FITTING
# ---------------------------
all_text = df["resume_text"] + " " + df["job_description"]

# ---------------------------
# FIT VECTORIZER ON FULL DATA (IMPORTANT)
# ---------------------------
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(all_text)

# ---------------------------
# BUILD FEATURES
# ---------------------------
X_features = []
y = df["label"].values

for i in range(len(df)):

    resume = df.iloc[i]["resume_text"]
    jd = df.iloc[i]["job_description"]

    tfidf = vectorizer.transform([resume, jd])

    cosine = cosine_similarity(tfidf[0], tfidf[1])[0][0]

    features = np.hstack([tfidf.toarray().flatten(), cosine])

    X_features.append(features)

X = np.array(X_features)

# ---------------------------
# TRAIN MODEL
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# SAVE
# ---------------------------
joblib.dump(model, "models/model_rf.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ RF trained + saved")