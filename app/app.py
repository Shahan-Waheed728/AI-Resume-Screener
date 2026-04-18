import streamlit as st
import os
import numpy as np
import joblib
import tensorflow as tf
import PyPDF2
import docx
from sklearn.metrics.pairwise import cosine_similarity
import time
import requests
import gdown
from datetime import datetime

def send_to_n8n(candidate_name, rf_result,
                ann_score, match_score, status):

    webhook_url = "https://shahan-waheed728.app.n8n.cloud/webhook/e852e99d-9807-4559-81f7-3a919da2bd65"

    payload = {
        "candidate_name": candidate_name,
        "rf_result": rf_result,
        "ml_ann_score": round(float(ann_score), 2),
        "ml_match_score": round(float(match_score), 2),
        "ml_status": status,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)        

        if response.status_code == 200:
            st.toast("Sent to n8n (Google Sheets updated)")
        else:
            st.warning(f"⚠️ n8n error: {response.status_code}")

    except Exception as e:
        st.warning(f"⚠️ n8n failed: {str(e)}")

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Resume Screener Pro",   
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #080c14; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Hide default streamlit header */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero Title ── */
    .hero-wrap {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4da3ff 0%, #a78bfa 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: 0.4rem;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Cards ── */
    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    /* ── Metric Boxes ── */
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.5rem 1rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        line-height: 1;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.4rem;
    }

    /* ── Score Colors ── */
    .score-green  { color: #34d399; }
    .score-yellow { color: #fbbf24; }
    .score-red    { color: #f87171; }
    .score-blue   { color: #60a5fa; }

    /* ── Status Badge ── */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-hire   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
    .badge-review { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
    .badge-reject { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

    /* ── Section Headers ── */
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #4da3ff;
        margin-bottom: 0.5rem;
    }

    /* ── Divider ── */
    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 1.5rem 0;
    }

    /* ── Progress Track ── */
    .progress-wrap { margin: 0.8rem 0; }
    .progress-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.3rem;
    }
    .progress-track {
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 99px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 1s ease;
    }
    .fill-blue   { background: linear-gradient(90deg, #4da3ff, #a78bfa); }
    .fill-green  { background: linear-gradient(90deg, #34d399, #4da3ff); }
    .fill-yellow { background: linear-gradient(90deg, #fbbf24, #f97316); }
    .fill-red    { background: linear-gradient(90deg, #f87171, #ec4899); }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #4da3ff, #a78bfa) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.75rem 2rem !important;
        letter-spacing: 0.05em !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(77,163,255,0.3) !important;
        border-radius: 12px !important;
        background: rgba(77,163,255,0.03) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODELS
# ---------------------------
# ---------------------------
# GOOGLE DRIVE MODEL LOADING
# ---------------------------
RF_MODEL_ID = "1TKgWtF8cTIj-WJJATxpWx-DQOyFAwOUy"
ANN_MODEL_ID = "1_0JEBm9b9o1RHWTKdt84gI4-ncXBcyVL"
TFIDF_ID = "1fY1l2Q4B-4d0yPeV1p35Q0FJoPrMXkmk"
VECTORIZER_ID = "1wanY6PX0yBqAiYW3voB-cY3RK-zsHcbh"

def download_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

@st.cache_resource
def load_models():

    rf_path = "models/model_rf.pkl"
    ann_path = "models/model_ann.h5"
    tfidf_path = "models/tfidf_vectorizer.pkl"
    vec_path = "models/vectorizer.pkl"

    download_from_drive(RF_MODEL_ID, rf_path)
    download_from_drive(ANN_MODEL_ID, ann_path)
    download_from_drive(TFIDF_ID, tfidf_path)
    download_from_drive(VECTORIZER_ID, vec_path)

    rf_model = joblib.load(rf_path)
    ann_model = tf.keras.models.load_model(ann_path, compile=False)
    vectorizer = joblib.load(vec_path)

    return rf_model, ann_model, vectorizer


rf_model, ann_model, vectorizer = load_models()

# ---------------------------
# FILE READERS
# ---------------------------
def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    return "".join([p.extract_text() or "" for p in pdf.pages])

def read_docx(file):
    doc = docx.Document(file)
    return " ".join([p.text for p in doc.paragraphs])

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
def build_features(resume, jd):
    tfidf  = vectorizer.transform([resume, jd])
    cosine = cosine_similarity(tfidf[0], tfidf[1])[0][0]
    features = np.hstack([tfidf.toarray().flatten(), cosine])
    return features.reshape(1, -1), cosine

# ---------------------------
# PROGRESS BAR HTML
# ---------------------------
def progress_bar(label, value_pct, fill_class):
    value_pct = max(0, min(100, value_pct))
    return f"""
    <div class="progress-wrap">
        <div class="progress-label">
            <span>{label}</span>
            <span>{value_pct:.1f}%</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill {fill_class}" style="width:{value_pct}%"></div>
        </div>
    </div>
    """

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.markdown('<p class="section-label">Recruiter Settings</p>', unsafe_allow_html=True)
    threshold = st.slider("Hire Threshold (%)", 0, 100, 60,
                          help="Candidates above this AI score will be shortlisted")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<p class="section-label">Threshold Guide</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#9ca3af; line-height:1.8;">
        🟢 <b style="color:#34d399">≥ threshold</b> — Hire / Shortlist<br>
        🟡 <b style="color:#fbbf24">≥ 40%</b> — Needs Review<br>
        🔴 <b style="color:#f87171">< 40%</b> — Reject
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Models Active</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#9ca3af; line-height:1.8;">
        ✅ Random Forest (Classifier)<br>
        ✅ ANN (Match Scorer)<br>
        ✅ TF-IDF + Cosine Similarity
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# HERO HEADER
# ---------------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">AI Resume Screener Pro</div>
    <div class="hero-sub">Intelligent candidate shortlisting · Powered by ML & Deep Learning</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ---------------------------
# INPUT SECTION
# ---------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Candidate Resume</p>', unsafe_allow_html=True)
    file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"],
                             label_visibility="collapsed")
    if file:
        st.success(f"✅ {file.name} uploaded successfully")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Job Description</p>', unsafe_allow_html=True)
    jd = st.text_area("Paste job description here", height=180,
                       placeholder="Paste the full job description here...",
                       label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# ANALYZE BUTTON
# ---------------------------
col_btn = st.columns([1, 2, 1])[1]
with col_btn:
    run = st.button("Run AI Screening", use_container_width=True)

# ---------------------------
# ANALYSIS
# ---------------------------
if run:
    if file and jd.strip():
        with st.spinner("Analyzing candidate profile..."):
            time.sleep(0.5)

            # Read resume
            resume = read_pdf(file) if file.name.endswith(".pdf") else read_docx(file)

            # Build features
            X, cosine = build_features(resume, jd)

            # Random Forest prediction
            rf_pred  = rf_model.predict(X)[0]
            rf_label = "Qualified" if rf_pred == 1 else "Not Qualified"
            rf_proba = rf_model.predict_proba(X)[0][1] * 100

            # ANN score — sigmoid output is already 0-1, multiply by 100
            ann_raw   = float(ann_model.predict(X, verbose=0)[0][0])
            ann_score = max(0.0, min(100.0, ann_raw * 100))  # Clamp 0-100
 
            st.write("Cosine:", cosine)
            st.write("ANN Score:", ann_score)

            # Cosine match score
            match_pct = cosine * 100
        #   Combined scoring — more reliable decision
            combined_score = (match_pct * 0.7) + (ann_score * 0.2) + (rf_proba * 0.1)

            if combined_score >= threshold:
                status      = "✅ HIRE / SHORTLIST"
                badge_class = "badge-hire"
                score_class = "score-green"
                fill_class  = "fill-green"
            elif combined_score >= 40:
                status      = "⚠️ UNDER CONSIDERATION"
                badge_class = "badge-review"
                score_class = "score-yellow"
                fill_class  = "fill-yellow"
            else:
                status      = "❌ REJECT"
                badge_class = "badge-reject"
                score_class = "score-red"
                fill_class  = "fill-red"
            # ---------------------------
            # SEND TO n8n (AFTER DECISION)
            # ---------------------------
            candidate_name = file.name if file else "Unknown"

            # optional: clean status text
            status_clean = status.replace("✅", "").replace("⚠️", "").replace("❌", "").strip()

            # send only meaningful results (recommended)
            if combined_score >= 40:
                send_to_n8n(
                    candidate_name,
                    rf_label,
                    ann_score,
                    combined_score,
                    status_clean
                )
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Candidate Analysis Dashboard</p>',
                    unsafe_allow_html=True)

        # ── Metric Cards ──
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🔗 Match Score</div>
                <div class="metric-value score-blue">{match_pct:.1f}%</div>
                <div class="metric-sub">Cosine Similarity</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AI Score</div>
                <div class="metric-value {score_class}">{ann_score:.1f}%</div>
                <div class="metric-sub">ANN Confidence</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RF Score</div>
                <div class="metric-value score-blue">{rf_proba:.1f}%</div>
                <div class="metric-sub">Random Forest</div>
            </div>""", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Decision</div>
                <div style="margin-top:0.6rem">
                    <span class="status-badge {badge_class}">{status}</span>
                </div>
                <div class="metric-sub">Threshold: {threshold}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Score Visualization ──
        col_left, col_right = st.columns([1.2, 0.8], gap="large")

        with col_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">Score Breakdown</p>',
                        unsafe_allow_html=True)
            st.markdown(progress_bar("Cosine Match Score", match_pct, "fill-blue"),
                        unsafe_allow_html=True)
            st.markdown(progress_bar("ANN AI Confidence", ann_score, fill_class),
                        unsafe_allow_html=True)
            st.markdown(progress_bar("Random Forest Probability", rf_proba, "fill-blue"),
                        unsafe_allow_html=True)
            st.markdown(progress_bar("Hire Threshold", threshold, "fill-yellow"),
                        unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">Decision Summary</p>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:0.9rem; color:#9ca3af; line-height:2.2;">
                <b style="color:#e5e7eb;">Random Forest:</b> {rf_label}<br>
                <b style="color:#e5e7eb;">ANN Score:</b> {ann_score:.2f}%<br>
                <b style="color:#e5e7eb;">Match Score:</b> {match_pct:.2f}%<br>
                <b style="color:#e5e7eb;">Threshold:</b> {threshold}%<br>
                <b style="color:#e5e7eb;">Final Decision:</b>
                <span class="status-badge {badge_class}" style="font-size:0.75rem;padding:0.3rem 0.8rem">{status}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.error("Please upload a resume AND paste a job description before screening.")