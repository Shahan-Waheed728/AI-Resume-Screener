import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

def compute_cosine(resume, jd):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume, jd])
    score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    return round(score * 100, 2)