# AI Resume Screener & Job Matcher 

An end-to-end **AI-powered recruitment system** that screens resumes, ranks candidates, and triggers automated hiring workflows using **Machine Learning, Deep Learning, and n8n automation**.

---

## Overview

Hiring teams deal with hundreds of resumes for every job posting. Manual screening is slow, biased, and inefficient.

This project solves this with a **dual-pipeline AI system** that:
- Screens resumes instantly
- Scores candidates using ML + Deep Learning
- Automates evaluation using AI workflows
- Logs results in real-time for recruiters

---

## Workflow

1. Upload resume and job description  
2. Apply NLP preprocessing  
3. Extract features using TF-IDF  
4. Evaluate using:
   - Random Forest (classification)
   - ANN (match scoring)
5. Generate final decision  
6. Trigger n8n workflow  
7. AI evaluation and summarization  
8. Store results in Google Sheets  

---

## Key Features

- Resume parsing (PDF & DOCX)
- NLP preprocessing (tokenization, stopword removal, lemmatization)
- TF-IDF + cosine similarity matching
- Random Forest classifier (~90% accuracy)
- ANN match scoring (0–100%)
- Adjustable hiring threshold
- n8n automation workflow
- AI-powered evaluation using GPT
- Google Sheets logging
- Streamlit deployment

---

## System Architecture

### 🔹 Pipeline 1 — ML Screening

<p align="center">
  <img src="images/ML Screening.png" width="350"/>
</p>

### 🔹 Pipeline 2 — AI Workflow (n8n)

<p align="center">
  <img src="images/AI-Workflow(n8n).png" width="350"/>
</p>

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python |
| ML | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| NLP | NLTK, TF-IDF |
| Frontend | Streamlit |
| Automation | n8n |
| AI | OpenAI GPT-4o-mini |
| Storage | Google Sheets, Drive |
| Version Control | GitHub |

---

## Project Structure
```bash
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
├── samples/
│   ├── Resume_HR_Fatima_Khan.pdf
│   ├── Resume_Finance_Omar_Sheikh.pdf
│   ├── Resume_Software_Ali_Hassan.pdf
│   └── Resume_Chef_Usman_Tariq.docx
├── n8n_workflow/
│   └── AI-Resume-Screener.json
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train_rf.py
│   ├── train_ann.py
│   ├── fix_dataset.py
│   └── build_dataset.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```
---


## Quick Test Guide (For Evaluators)

### Steps

| Step | Action | What to Do |
|------|--------|-----------|
| 1️⃣ | Open App | Visit deployed app |
| 2️⃣ | Download Resume | Use sample resumes in app |
| 3️⃣ | Load JD | Select from sample JD dropdown |
| 4️⃣ | Run Screening | Upload resume → Click "Run AI Screening" |

### Expected Results

| Resume File | Job Role | Expected Output |
|------------|---------|----------------|
| Resume_HR_Fatima_Khan | HR Manager | ✅ HIRE |
| Resume_Finance_Omar_Sheikh | Finance Analyst | ⚠️ UNDER CONSIDERATION |
| Resume_Software_Ali_Hassan | Software Engineer | ⚠️ UNDER CONSIDERATION |
| Resume_Chef_Usman_Tariq | Tech/Finance JD | ❌ REJECT |

**Note:** Default threshold = 60%

---

## Local Setup Instructions

```bash
git clone https://github.com/Shahan-Waheed728/AI-Resume-Screener.git
cd AI-Resume-Screener

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app/app.py
```
---

## Usage Guide

### Steps to Use

1. Open the application  
2. Upload a resume (**PDF/DOCX**)  
3. Paste or select a job description  
4. Click **Run AI Screening**  

### Output

After processing, you will see:
- **Match Score**
- **AI Evaluation**
- **Final Decision**

---

## Deployment (Streamlit)

### Steps

1. Push your code to GitHub  
2. Go to https://share.streamlit.io  
3. Connect your repository  
4. Set the main file path: 
   app/app.py 
5. Add environment variable:
   N8N_WEBHOOK_URL = https://shahan-waheed728.app.n8n.cloud/webhook/e852e99d-9807-4559-81f7-3a919da2bd65
6. Click **Deploy**

---

## Troubleshooting

| Issue | Solution |
|------|---------|
| App not loading | Check Streamlit logs |
| Model not found | Ensure model files exist |
| NLTK error | Run nltk downloads |
| n8n not working | Verify webhook URL |

---

## Future Enhancements

- BERT embeddings for better semantic matching  
- Named Entity Recognition (NER) for skill extraction  
- Real-time job scraping integration  
- Recruiter feedback loop  
- Multi-language support  
- Interview prediction system  

---

## 🔗 Live Demo & Links

- **Deployed App:** https://ai-resume-screener-drymmejxbnd7uguuztxwpt.streamlit.app  
- **GitHub Repo:** https://github.com/Shahan-Waheed728/AI-Resume-Screener  
- **Medium Article:** (Add your link here)  
- **LinkedIn:** (Add your profile link)

---

## Disclaimer

This project was developed as a **final project for an AI/ML Fellowship** and was completed as a **solo project**.

It is intended for **educational and demonstration purposes only**. The system is not designed to replace human judgment in real-world hiring decisions and should not be used as the sole evaluation tool for candidates.

---

## Acknowledgement

- Open-source community  
- Scikit-learn & TensorFlow contributors  
- n8n platform  
- Dataset providers  

---

## Author

**Shahan Waheed**  
Software Engineering Student (7th Semester)  
COMSATS University Attock  

---

## License

MIT License — free to use with attribution.