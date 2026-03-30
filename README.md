# 🚀 AI Resume Screener & Job Matcher

An AI-powered system that automates resume screening, ranks candidates, and matches them with job descriptions using Machine Learning and Deep Learning.

## 📖 Project Overview

Hiring teams often deal with hundreds of resumes. This project automates the screening process using NLP and AI models to:

- Extract and preprocess resume & job description text
- Classify candidates (Qualified / Not Qualified)
- Rank candidates based on job match score (0–100)
- Trigger automated recruiter workflows

## 🎯 Features

- NLP-based Resume & Job Description Processing
- TF-IDF + Cosine Similarity
- Random Forest Classifier (ML)
- ANN Model for Candidate Ranking (DL)
- Streamlit Web Application
- n8n Automation Integration
- Candidate Score Threshold System

## 🧠 Tech Stack

- Python
- Scikit-learn
- TensorFlow / Keras
- NLP (TF-IDF, Cosine Similarity)
- Streamlit
- n8n (Automation)
- Pandas, NumPy

## 🏗️ Project Architecture
Resume + JD → NLP Processing → Feature Engineering → RF Classifier + ANN Model → Score & Ranking → Streamlit UI → n8n Automation

## 📊 Model Details

- Random Forest → Classification (Qualified / Not Qualified)
- ANN → Match Score Prediction (0–100)
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC

## 📁 Dataset

- Resume Dataset (Kaggle)
- Job Description Dataset (Kaggle)

## 🚧 Project Status

🟡 In Development (Phase 1: Data Processing & Model Building)

## 🌐 Future Improvements

- Multi-language resume support
- Real-time job scraping
- Interview prediction system

## 👨‍💻 Author

**Shahan Waheed**  
AI/ML Fellowship – Solo Project

