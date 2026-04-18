import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load data
resume_df = pd.read_csv("data/resume_preprocessed.csv")
jd_df = pd.read_csv("data/jobs_preprocessed.csv")
vectorizer = joblib.load("models/vectorizer.pkl")

print("Building correct dataset...")
rows = []

for i in range(len(resume_df)):
    resume_text = str(resume_df.iloc[i]['processed_text'])
    category    = resume_df.iloc[i]['category']

    # ── Positive example: matching category JD ──
    mask = jd_df['job_title'].str.contains(
        category.replace('-', ' '), case=False, na=False
    )
    matched = jd_df[mask]

    if len(matched) > 0:
        # Pick best matching JD from same category
        pos_jd = matched.sample(1).iloc[0]
        rows.append({
            'resume_text'    : resume_text,
            'job_description': str(pos_jd['combined_text']),
            'job_title'      : pos_jd['job_title'],
            'category'       : category,
            'label'          : 1   # ✅ Same category = Qualified
        })

    # ── Negative example: completely different category JD ──
    not_mask = ~jd_df['job_title'].str.contains(
        category.replace('-', ' '), case=False, na=False
    )
    not_matched = jd_df[not_mask]

    if len(not_matched) > 0:
        neg_jd = not_matched.sample(1).iloc[0]
        rows.append({
            'resume_text'    : resume_text,
            'job_description': str(neg_jd['combined_text']),
            'job_title'      : neg_jd['job_title'],
            'category'       : category,
            'label'          : 0   # ❌ Different category = Not Qualified
        })

df = pd.DataFrame(rows)

print(f"\n✅ Dataset shape: {df.shape}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nSample correct pairs:")
print(df[['category', 'job_title', 'label']].head(10))

df.to_csv("data/final_dataset.csv", index=False)
print("\n✅ Correct dataset saved!")