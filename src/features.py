import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import json

# ── Base directory (project root) ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = os.path.join(BASE_DIR, "data")
models_dir = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


def build_tfidf_features(resume_df, jd_df):
    """
    Build TF-IDF features from resume and JD text.
    """
    print("Building TF-IDF features...")

    # Combine all text
    all_text = pd.concat([
        resume_df['processed_text'],
        jd_df['combined_text']
    ]).fillna('')

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    tfidf.fit(all_text)

    # Transform (KEEP SPARSE → memory efficient)
    resume_vectors = tfidf.transform(resume_df['processed_text'].fillna(''))
    jd_vectors = tfidf.transform(jd_df['combined_text'].fillna(''))

    print(f"Resume vectors shape: {resume_vectors.shape}")
    print(f"JD vectors shape: {jd_vectors.shape}")

    # Save vectorizer
    with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    print("✅ TF-IDF vectorizer saved")

    return tfidf, resume_vectors, jd_vectors


def create_labeled_dataset(resume_df, jd_df):
    """
    Create labeled dataset using cosine similarity.
    """
    print("\nCreating labeled dataset...")

    tfidf, resume_vectors, jd_vectors = build_tfidf_features(resume_df, jd_df)

    print("\nComputing similarity scores...")
    scores = []
    best_jd_titles = []

    for i, rv in enumerate(resume_vectors):

        category = resume_df.iloc[i]['category']

        mask = jd_df['job_title'].str.contains(
            str(category).replace('-', ' '),
            case=False,
            na=False
        )

        matched_jd_indices = jd_df[mask].index.tolist()

        if matched_jd_indices:
            matched_vectors = jd_vectors[matched_jd_indices]
            sim = cosine_similarity(rv, matched_vectors)
            best_score = float(np.max(sim))
            best_idx = matched_jd_indices[int(np.argmax(sim))]
        else:
            sim = cosine_similarity(rv, jd_vectors)
            best_score = float(np.max(sim))
            best_idx = int(np.argmax(sim))

        best_title = jd_df.iloc[best_idx]['job_title']

        scores.append(round(best_score * 100, 2))
        best_jd_titles.append(best_title)

        if (i + 1) % 500 == 0:
           print(f"  Processed {i + 1}/{resume_vectors.shape[0]} resumes...")

    # Add results
    resume_df = resume_df.copy()
    resume_df['match_score'] = scores
    resume_df['best_match_job'] = best_jd_titles

    # Dynamic thresholds
    q75 = resume_df['match_score'].quantile(0.75)
    q50 = resume_df['match_score'].quantile(0.50)

    print(f"\nUsing dynamic thresholds:")
    print(f"  ✅ Qualified     : >= {q75:.2f}")
    print(f"  ⚠️  Review        : >= {q50:.2f}")
    print(f"  ❌ Not Qualified  : <  {q50:.2f}")

    resume_df['label'] = resume_df['match_score'].apply(
        lambda x: 1 if x >= q75 else 0
    )

    resume_df['status'] = resume_df['match_score'].apply(
        lambda x: 'Qualified' if x >= q75
        else ('Review' if x >= q50 else 'Not Qualified')
    )

    # Save thresholds
    thresholds = {'qualified': float(q75), 'review': float(q50)}
    with open(os.path.join(models_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f)

    print("✅ Thresholds saved")

    # Save dataset
    resume_df.to_csv(os.path.join(data_dir, "labeled_dataset.csv"), index=False)

    print(f"\n✅ Labeled dataset saved! Shape: {resume_df.shape}")
    print("\nLabel distribution:")
    print(resume_df['status'].value_counts())

    return resume_df, tfidf, resume_vectors, jd_vectors


def prepare_model_features(resume_df, resume_vectors):
    """
    Prepare final feature matrix X and labels y.
    """
    print("\nPreparing feature matrix...")

    match_scores = resume_df['match_score'].values.reshape(-1, 1)

    # Convert sparse → dense ONLY here (controlled)
    X = np.hstack([resume_vectors.toarray(), match_scores])
    y = resume_df['label'].values

    print(f"✅ Feature matrix shape: {X.shape}")
    print(f"✅ Labels shape: {y.shape}")

    # Save features
    np.save(os.path.join(data_dir, "X_features.npy"), X)
    np.save(os.path.join(data_dir, "y_labels.npy"), y)

    print("✅ Features saved")

    return X, y


if __name__ == "__main__":

    print("Loading datasets...")

    resume_path = os.path.join(data_dir, "resume_preprocessed.csv")
    jd_path = os.path.join(data_dir, "jobs_preprocessed.csv")

    resume_df = pd.read_csv(resume_path)
    jd_df = pd.read_csv(jd_path)

    # Create labeled dataset
    labeled_df, tfidf, resume_vecs, jd_vecs = create_labeled_dataset(
        resume_df, jd_df
    )

    # Prepare features
    X, y = prepare_model_features(labeled_df, resume_vecs)

    print("\n🎉 Feature engineering COMPLETE!")