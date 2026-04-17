import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and normalize raw text."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters & numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    """Remove stopwords from text."""
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def lemmatize_text(text):
    """Lemmatize words in text."""
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

def preprocess(text):
    """Full preprocessing pipeline."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def preprocess_datasets():
    """Load, preprocess and save both datasets."""

    print("Loading datasets...")
    resume = pd.read_csv("data/resume_cleaned.csv")
    jd = pd.read_csv("data/jobs_cleaned.csv")

    print("Preprocessing resume text...")
    resume['processed_text'] = resume['resume_text'].apply(preprocess)

    print("Preprocessing job description text...")
    jd['processed_text'] = jd['job_description'].apply(preprocess)

    # Combine skills + qualifications into JD processed text
    jd['combined_text'] = (
        jd['processed_text'] + ' ' +
        jd['skills'].apply(preprocess) + ' ' +
        jd['qualifications'].apply(preprocess)
    )

    # Save preprocessed datasets
    resume.to_csv("data/resume_preprocessed.csv", index=False)
    jd.to_csv("data/jobs_preprocessed.csv", index=False)

    print(f"\n✅ Resume preprocessed! Shape: {resume.shape}")
    print(f"✅ JD preprocessed! Shape: {jd.shape}")
    print("\nSample preprocessed resume:")
    print(resume['processed_text'].iloc[0][:200])
    print("\nSample preprocessed JD:")
    print(jd['combined_text'].iloc[0][:200])

    return resume, jd

if __name__ == "__main__":
    preprocess_datasets()