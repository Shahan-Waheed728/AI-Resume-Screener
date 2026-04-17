import pandas as pd

# Load datasets
resume_df = pd.read_csv("data/resume_preprocessed.csv")
job_df = pd.read_csv("data/job_descriptions.csv")

# Use cleaned resume text
resume_df = resume_df.dropna(subset=["processed_text"])
job_df = job_df.dropna(subset=["Job Description"])

data = []

for i in range(len(resume_df)):

    resume = resume_df.iloc[i]["processed_text"]

    # Random job (positive)
    job_match = job_df.sample(1).iloc[0]["Job Description"]

    # Another random job (negative)
    job_mismatch = job_df.sample(1).iloc[0]["Job Description"]

    # Add samples
    data.append([resume, job_match, 1])
    data.append([resume, job_mismatch, 0])

# Create final dataset
final_df = pd.DataFrame(data, columns=[
    "resume_text", "job_description", "label"
])

# Save
final_df.to_csv("data/final_dataset.csv", index=False)

print("✅ Final dataset created successfully")