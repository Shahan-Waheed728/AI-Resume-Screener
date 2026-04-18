import pandas as pd

# ── Job Description Dataset ──
jd = pd.read_csv("data/job_descriptions.csv")

keep_cols = ['Job Title', 'Job Description', 'skills',
             'Qualifications', 'Experience', 'Role']
jd = jd[keep_cols]
jd = jd.dropna()
jd = jd.sample(n=5000, random_state=42)
jd.columns = ['job_title', 'job_description',
              'skills', 'qualifications',
              'experience', 'role']
jd.to_csv("data/jobs_cleaned.csv", index=False)
print(f"JD Dataset cleaned! Shape: {jd.shape}")

# ── Resume Dataset ──
resume = pd.read_csv("data/Resume.csv")

# ✅ Fixed: Capital R in Resume_str
resume = resume[['Resume_str', 'Category']]
resume = resume.dropna()
resume.columns = ['resume_text', 'category']

# ✅ Fixed: Save separately, never overwrite original!
resume.to_csv("data/resume_cleaned.csv", index=False)
print(f"\nResume Dataset cleaned! Shape: {resume.shape}")
print(resume.head(2))
print(f"\nResume categories:\n{resume['category'].unique()}")