AI Resume Screener & Job Matcher
An end-to-end AI-powered recruitment system that screens resumes, ranks candidates, and triggers automated hiring workflows using Machine Learning, Deep Learning, and n8n automation.

Live Demo

🚀 Deployed App: AI Resume Screener Pro
🔗 GitHub Repo: Shahan-Waheed728/AI-Resume-Screener
📝 Medium Article: How I Built an AI Resume Screener


Project Overview
Hiring teams deal with hundreds of resumes for every job posting. Manual screening is slow, biased, and inefficient. This project solves this with a dual-pipeline AI system that screens candidates instantly and logs all results automatically.

System Architecture
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    PIPELINE 1 — ML SCREENING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   [Resume PDF/DOCX]        [Job Description Text]
          │                          │
          └──────────┬───────────────┘
                     ▼
          ┌─────────────────────┐
          │  NLP Preprocessing  │
          │  • Tokenization     │
          │  • Stopword Removal │
          │  • Lemmatization    │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │  Feature Engineering│
          │  • TF-IDF Vectors   │
          │  • Cosine Similarity│
          │  • Feature Matrix   │
          └──────────┬──────────┘
                     ▼
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
 ┌────────────────┐   ┌────────────────┐
 │ Random Forest  │   │   ANN Model    │
 │  Classifier    │   │    Scorer      │
 │                │   │                │
 │ Qualified /    │   │  Match Score   │
 │ Not Qualified  │   │   0 – 100%     │
 └───────┬────────┘   └───────┬────────┘
         └──────────┬──────────┘
                    ▼
         ┌──────────────────────┐
         │    Final Decision    │
         │  ✅ HIRE  >= 60%     │
         │  ⚠️  REVIEW  >= 40%  │
         │  ❌ REJECT  < 40%    │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  n8n Webhook POST    │
         │  → Google Sheets     │
         │    ML_Results Tab    │
         └──────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  PIPELINE 2 — AI WORKFLOW (n8n)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Form: Full-Stack Dev]    [Form: Account Assistant]
             │                          │
             └──────────┬───────────────┘
                        ▼
             ┌─────────────────────┐
             │   Google Drive      │
             │   Resume Upload     │
             └──────────┬──────────┘
                        ▼
             ┌─────────────────────┐
             │  Extract from File  │
             └──────────┬──────────┘
                        ▼
          ┌─────────────┴─────────────┐
          ▼                           ▼
 ┌────────────────┐         ┌────────────────────┐
 │ Extract        │         │ Extract             │
 │ Personal Info  │         │ Professional Info   │
 │ • Name, Email  │         │ • Skills            │
 │ • Phone        │         │ • Experience        │
 │ • Location     │         │ • Education         │
 └───────┬────────┘         └─────────┬──────────┘
         └──────────┬─────────────────┘
                    ▼
         ┌──────────────────────┐
         │  OpenAI GPT-4o-mini  │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  Summarization Chain │
         │  400-word profile    │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │  Evaluate & Score    │
         │  • ai-resume-score   │
         │  • ai-evaluations    │
         └──────────┬───────────┘
                    ▼
              [Switch Node]
          ┌────────┼────────┐
          ▼        ▼        ▼
   [Shortlisted] [Maybe] [Ignored]
          └────────┼────────┘
                   ▼
         ┌──────────────────────┐
         │    Google Sheets     │
         │  Full Profile Log    │
         └──────────────────────┘

Features

Resume parsing — supports PDF and DOCX formats
NLP pipeline — tokenization, stopword removal, lemmatization
TF-IDF vectorization with cosine similarity matching
Random Forest classifier — Qualified / Not Qualified
ANN match scorer — candidate match score 0–100%
Adjustable hire threshold via recruiter sidebar
n8n automation — GPT-powered candidate profiling
Google Sheets logging — all results auto-logged
Deployed live on Streamlit Cloud


Model Performance
Random Forest Classifier
MetricScoreAccuracy89.92%Precision82.89%Recall82.35%F1 Score82.62%CV Mean F181.89% ± 2.6%Test Samples526
Confusion Matrix:
                   Predicted
              Not Qualified  |  Qualified
Actual Not Q      347        |     26
Actual Qualified   27        |    126
ANN Model (Match Scorer)
MetricValueArchitecture512 → 256 → 128 → 64 → 1Output ActivationSigmoid (0–1)Loss FunctionBinary CrossentropyOptimizerAdam (lr=0.001)RegularizationBatchNorm + Dropout (0.4, 0.3, 0.2)Training StrategyEarly Stopping (patience=5)Output Range0–100% match score

Note: The ANN acts as a regression scorer, not a classifier. It outputs a continuous match probability between 0 and 1, displayed as a percentage score to the recruiter.


Tech Stack
CategoryTechnologyLanguagePython 3.13ML ModelScikit-learn — Random ForestDeep LearningTensorFlow 2.21 / Keras 3.14NLPNLTK, TF-IDF, Cosine SimilarityWeb AppStreamlit 1.56Automationn8n CloudAI / LLMOpenAI GPT-4o-miniStorageGoogle Sheets, Google DriveDeploymentStreamlit CloudVersion ControlGit / GitHub

Project Structure
AI-Resume-Screener/
├── app/
│   └── app.py                  Streamlit web application
├── data/
│   ├── final_dataset.csv       Labeled training dataset (3501 rows)
│   └── clean_data.py           Data cleaning script
├── models/
│   ├── model_rf.pkl            Trained Random Forest model
│   ├── model_ann.h5            Trained ANN model
│   ├── tfidf_vectorizer.pkl    TF-IDF vectorizer
│   ├── vectorizer.pkl          Feature vectorizer
│   └── thresholds.json         Dynamic scoring thresholds
├── src/
│   ├── preprocess.py           NLP preprocessing pipeline
│   ├── features.py             Feature engineering
│   ├── train_rf.py             Random Forest training script
│   ├── train_ann.py            ANN training script
│   ├── fix_dataset.py          Dataset correction script
│   └── build_dataset.py        Dataset builder
├── .streamlit/
│   └── config.toml             Streamlit configuration
├── requirements.txt            Project dependencies
└── README.md                   Project documentation

Dataset
DatasetSourceRowsPurposeResume DatasetKaggle2,484TrainingJob Description DatasetKaggle5,000 (sampled)TrainingFinal Labeled DatasetCategory-aware generation3,501 pairsModel Training
24 resume categories: HR, Engineering, Finance, IT, Healthcare, Sales, Marketing, Education, Banking, Agriculture, and more.

Run Locally
bash# Clone the repository
git clone https://github.com/Shahan-Waheed728/AI-Resume-Screener.git
cd AI-Resume-Screener

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Run the app
streamlit run app/app.py

How to Use

Open the app at localhost:8501 or the live link
Set hire threshold using the sidebar slider (default: 60%)
Upload a resume in PDF or DOCX format
Paste the job description in the text area
Click Run AI Screening
View the results — Match Score, AI Score, RF Score, Final Decision
Results are automatically logged to Google Sheets


Deployment
bash# Push to GitHub
git add .
git commit -m "final: complete AI Resume Screener ready for deployment"
git push origin main
Then on Streamlit Cloud:

Go to share.streamlit.io
Connect GitHub repo — Shahan-Waheed728/AI-Resume-Screener
Set main file path — app/app.py
Add secret — N8N_WEBHOOK_URL = your-production-webhook-url
Click Deploy


Future Improvements

BERT embeddings for semantic matching beyond keyword overlap
Named Entity Recognition for automatic skills extraction
Real-time job scraping from LinkedIn and Indeed via n8n
Recruiter feedback loop to retrain models automatically
Multi-language support for Urdu and Arabic resumes
Interview performance prediction system


Author
Shahan Waheed
7th Semester, Software Engineering — COMSATS University Attock
AI/ML Fellowship — Solo Project — 2026

License
This project is licensed under the MIT License. You are free to use, copy, modify, and distribute this project as long as you include the original author credit. See the LICENSE file for details.