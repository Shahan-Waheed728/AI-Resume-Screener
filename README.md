# AI Resume Screener & Job Matcher

An end-to-end **AI-powered recruitment system** that screens resumes, ranks candidates, and triggers automated hiring workflows using **Machine Learning, Deep Learning, and n8n automation**.

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

## Live Demo & Links

* **Deployed App:** https://ai-resume-screener-drymmejxbnd7uguuztxwpt.streamlit.app
* **GitHub Repo:** [https://github.com/Shahan-Waheed728/AI-Resume-Screener](https://github.com/Shahan-Waheed728/AI-Resume-Screener)
* **Medium Article:** How I Built an AI Resume Screener
* **LinkedIn:** Shahan Waheed

---

## Quick Test Guide (For Evaluators)

Follow these steps to test the system in **under 2 minutes**:

### 🪜 Steps

| Step | Action | What to Do |
|------|--------|-----------|
| 1️⃣ | Open App | Visit the deployed app link |
| 2️⃣ | Download Resume | Click **"Download Sample Resumes"** inside the app |
| 3️⃣ | Load Job Description | Select from **"Sample Job Descriptions"** dropdown |
| 4️⃣ | Run Screening | Upload resume → Click **"Run AI Screening"** |

---

### Expected Results

| Resume File | Job Role | Expected Output |
|------------|---------|----------------|
| Resume_HR_Fatima_Khan | HR Manager | ✅ HIRE |
| Resume_Finance_Omar_Sheikh | Finance Analyst | ⚠️ UNDER CONSIDERATION |
| Resume_Software_Ali_Hassan | Software Engineer | ⚠️ UNDER CONSIDERATION |
| Resume_Chef_Usman_Tariq | Tech/Finance JD | ❌ REJECT |

**Note:** Default threshold is **60%**. You can adjust it using the sidebar slider.

---


## Project Overview

Hiring teams deal with hundreds of resumes for every job posting. Manual screening is slow, biased, and inefficient. This project solves this with a **dual-pipeline AI system** that screens candidates instantly and logs all results automatically.

---

## Features

* Resume parsing — supports PDF and DOCX formats
* NLP pipeline — tokenization, stopword removal, lemmatization
* TF-IDF vectorization with cosine similarity matching
* Random Forest classifier — Qualified / Not Qualified (~90% accuracy)
* ANN match scorer — candidate match score (0–100%)
* Adjustable hire threshold via recruiter sidebar
* n8n automation — GPT-powered candidate profiling workflow
* Google Sheets logging — real-time results tracking
* Deployed on Streamlit Cloud

---

## Model Performance

### Random Forest Classifier

| Metric     | Score         |
| ---------- | ------------- |
| Accuracy   | 89.92%        |
| Precision  | 82.89%        |
| Recall     | 82.35%        |
| F1 Score   | 82.62%        |
| CV Mean F1 | 81.89% ± 2.6% |

### ANN Model (Match Scorer)

* Architecture: 512 → 256 → 128 → 64 → 1
* Activation: Sigmoid (0–1)
* Loss: Binary Crossentropy
* Optimizer: Adam (lr=0.001)
* Regularization: BatchNorm + Dropout

> Note: ANN works as a regression-style scorer producing a continuous match percentage.

---

## Tech Stack

* **Language:** Python 3
* **ML:** Scikit-learn (Random Forest)
* **Deep Learning:** TensorFlow / Keras
* **NLP:** NLTK, TF-IDF, Cosine Similarity
* **Frontend:** Streamlit
* **Automation:** n8n
* **AI:** OpenAI GPT-4o-mini
* **Storage:** Google Sheets, Google Drive
* **Version Control:** Git + GitHub

---

## Project Structure

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

## ▶Run Locally

```bash
git clone https://github.com/Shahan-Waheed728/AI-Resume-Screener.git
cd AI-Resume-Screener

python -m venv venv
venv\\Scripts\\activate

pip install -r requirements.txt

streamlit run app/app.py
```

---

## Author

**Shahan Waheed**
Software Engineering Student (7th Semester)
COMSATS University Attock

---

## License

MIT License — free to use with attribution.
