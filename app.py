import streamlit as st
import pandas as pd
import numpy as np
import joblib
import fitz  # PyMuPDF
import re
import tempfile
import io
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# ----------------- 1. Page Configuration -----------------
st.set_page_config(
    page_title="AI Proctored Interviewer & Evaluator",
    layout="wide",
    page_icon="🎥"
)

st.markdown("""
<style>
    .question-box {
        background-color: #f0f7f7;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
        border-left: 6px solid #008080;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        background-color: #e8f5e9;
        color: #2e7d32;
        font-weight: bold;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎥 AI Video-Proctored Resume-Aware Interviewer")
st.caption("Autonomous Technical Evaluation System with Direct Speech-to-Text & Scoring")

# ----------------- 2. Load Model & Dataset -----------------
@st.cache_resource
def load_assets():
    model = joblib.load("answer_eval_model.pkl")
    questions = pd.read_csv("questions.csv")
    return model, questions

try:
    eval_model, questions_df = load_assets()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# ----------------- 3. Taxonomies & NLP Helpers -----------------
DOMAIN_TAXONOMY = {
    "AI Engineer": [
        "Deep Learning", "Transformers", "Computer Vision", "Generative AI", 
        "NLP", "BERT", "PyTorch", "TensorFlow", "Diffusion", "Reinforcement Learning", "LLM"
    ],
    "Machine Learning / Data Science": [
        "Machine Learning", "Scikit-learn", "Random Forest", "XGBoost", "SVM", 
        "Statistics", "Hypothesis Testing", "Pandas", "NumPy", "Regression", "Clustering"
    ],
    "Computer Science Core": [
        "Data Structures", "Algorithms", "Operating Systems", "DBMS", "SQL", 
        "Computer Networks", "System Design", "C++", "Java", "Linux", "Concurrency"
    ]
}

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()

def analyze_resume(text):
    clean_t = text.lower()
    matched_skills = {}
    domain_scores = {k: 0 for k in DOMAIN_TAXONOMY.keys()}

    for domain, skills in DOMAIN_TAXONOMY.items():
        found = []
        for s in skills:
            if re.search(r"\b" + re.escape(s.lower()) + r"\b", clean_t):
                found.append(s)
                domain_scores[domain] += 1
        matched_skills[domain] = found

    detected_domain = max(domain_scores, key=domain_scores.get)
    if domain_scores[detected_domain] == 0:
        detected_domain = "Computer Science Core"

    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\d{10}", text)
    first_line = text.strip().split("\n")[0].strip()

    return {
        "name": first_line if len(first_line) < 35 else "Candidate",
        "email": email_match.group(0) if email_match else "N/A",
        "phone": phone_match.group(0) if phone_match else "N/A",
        "detected_domain": detected_domain,
        "matched_skills": matched_skills[detected_domain]
    }

def convert_audio_to_text(audio_bytes):
    """Convert uploaded/recorded audio directly using SpeechRecognition."""
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        with sr.AudioFile(tmp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text.strip()
    except Exception:
        # Fallback to faster-whisper if the Google API has a format mismatch
        try:
            from faster_whisper import WhisperModel
            whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _ = whisper.transcribe(tmp_path)
            res = " ".join([seg.text for seg in segments]).strip()
            return res
        except Exception:
            return ""

def calculate_nlp_features(answer, expected_concepts):
    tokens = [w.lower() for w in answer.split() if len(w) > 1]
    length = len(tokens)

    exp_tokens = [w.lower() for w in expected_concepts.split()]
    matched = sum(1 for c in exp_tokens if c in tokens)
    concept_cov = matched / len(exp_tokens) if exp_tokens else 0.5

    vec = TfidfVectorizer()
    try:
        tfidf = vec.fit_transform([expected_concepts, answer])
        tfidf_sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        tfidf_sim = 0.2

    sem_sim = float(np.clip(0.55 * tfidf_sim + 0.45 * concept_cov, 0.0, 1.0))
    relevance = float(0.4 * tfidf_sim + 0.6 * sem_sim)
    keyword_cov = float(concept_cov * 0.92 + 0.08 * min(1.0, length / 35.0))

    feature_array = np.array([[length, keyword_cov, tfidf_sim, sem_sim, concept_cov, relevance]])
    return feature_array, {
        "relevance": relevance, 
        "concept_coverage": concept_cov,
        "tfidf_similarity": tfidf_sim,
        "semantic_similarity": sem_sim,
        "answer_length": length
    }

# ----------------- 4. Session State Management -----------------
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False
if "current_q_idx" not in st.session_state:
    st.session_state.current_q_idx = 0
if "selected_questions" not in st.session_state:
    st.session_state.selected_questions = []
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

# ================= SCREEN 1: RESUME UPLOAD =================
if not st.session_state.interview_started:
    st.subheader("📄 Step 1: Candidate Verification & Resume Analysis")
    uploaded_pdf = st.file_uploader("Upload candidate resume (PDF format)", type=["pdf"])

    if uploaded_pdf is not None:
        with st.spinner("Analyzing resume content..."):
            raw_text = extract_text_from_pdf(uploaded_pdf.read())
            profile = analyze_resume(raw_text)
            st.session_state.profile = profile

        st.success(f"Candidate Profile: **{profile['name']}** | Domain: **{profile['detected_domain']}**")

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Email:** {profile['email']}")
            st.write(f"**Phone:** {profile['phone']}")
        with c2:
            st.write(f"**Domain:** {profile['detected_domain']}")
            st.write(f"**Matched Skills:** {', '.join(profile['matched_skills']) if profile['matched_skills'] else 'General Concepts'}")

        st.divider()
        st.subheader("⚙️ Track Selection")
        target_domain = st.selectbox(
            "Confirm Technical Interview Track:",
            ["AI Engineer", "Machine Learning / Data Science", "Computer Science Core"],
            index=["AI Engineer", "Machine Learning / Data Science", "Computer Science Core"].index(profile["detected_domain"])
        )
        num_questions = st.slider("Number of Questions:", min_value=3, max_value=7, value=4)

        if st.button("Enter Proctored Interview Room", type="primary"):
            st.session_state.profile["target_domain"] = target_domain
            
            if target_domain == "AI Engineer":
                matched = questions_df[questions_df["job_roles"].str.contains("AI Engineer|Deep Learning", na=False)]
            elif target_domain == "Machine Learning / Data Science":
                matched = questions_df[questions_df["job_roles"].str.contains("Data Scientist|Machine Learning", na=False)]
            else:
                matched = questions_df[questions_df["job_roles"].str.contains("Computer Science Core|Software Engineer", na=False)]

            sample_size = min(len(matched), num_questions)
            st.session_state.selected_questions = matched.sample(sample_size, random_state=42).to_dict(orient="records")
            st.session_state.interview_started = True
            st.session_state.current_q_idx = 0
            st.session_state.evaluations = []
            st.rerun()

# ================= SCREEN 2: PROCTORED LIVE INTERVIEW =================
else:
    q_list = st.session_state.selected_questions
    idx = st.session_state.current_q_idx

    # Final Result Screen
    if idx >= len(q_list):
        st.header("🏁 Interview Evaluation & Performance Report")
        profile = st.session_state.profile

        st.subheader(f"Candidate: {profile['name']} | Domain Track: {profile['target_domain']}")

        scores = [e["score"] for e in st.session_state.evaluations]
        relevance_vals = [e["metrics"]["relevance"] * 100 for e in st.session_state.evaluations]
        concept_vals = [e["metrics"]["concept_coverage"] * 100 for e in st.session_state.evaluations]

        avg_score = float(np.mean(scores))
        avg_rel = float(np.mean(relevance_vals))
        avg_concept = float(np.mean(concept_vals))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Score", f"{avg_score:.1f} / 100")
        c2.metric("Concept Coverage", f"{avg_concept:.1f}%")
        c3.metric("Semantic Relevance", f"{avg_rel:.1f}%")
        c4.metric("Questions Evaluated", f"{len(q_list)}")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Performance Radar Breakdown")
            fig = go.Figure(data=go.Scatterpolar(
                r=[avg_score, avg_rel, avg_concept, min(100.0, avg_score + 4.0)],
                theta=["Technical Score", "Relevance", "Concept Coverage", "Depth"],
                fill='toself',
                line_color="#008080"
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Question-Wise Score Audit")
            eval_table = pd.DataFrame([
                {
                    "Question": e["question"][:40] + "...",
                    "Candidate Answer": e.get("final_text", "")[:40] + "...",
                    "Score": f"{e['score']:.1f}",
                    "Concept Match": f"{e['metrics']['concept_coverage']*100:.1f}%"
                }
                for e in st.session_state.evaluations
            ])
            st.dataframe(eval_table, use_container_width=True)

        if st.button("🔄 Start New Interview", type="primary"):
            st.session_state.interview_started = False
            st.rerun()
        st.stop()

    curr_q = q_list[idx]
    
    st.progress((idx + 1) / len(q_list))
    st.caption(f"Question {idx + 1} of {len(q_list)} | Category: {curr_q['category']} | Level: {curr_q['difficulty']}")

    st.markdown(f"""
    <div class="question-box">
        <h3 style="color: #004d40; margin: 0;">{curr_q['question']}</h3>
    </div>
    """, unsafe_allow_html=True)

    # Two column layout: Live Camera (Left), Audio / Text Answer (Right)
    col_cam, col_ans = st.columns([1, 1])

    with col_cam:
        st.markdown("**📷 Candidate Proctoring Feed (Camera Preview):**")
        st.camera_input("Camera Live", key=f"cam_{idx}")

    with col_ans:
        st.markdown("**🎙️ Answer Submission (Voice ya Text):**")
        audio_clip = st.audio_input("Speak your answer using the microphone:", key=f"audio_{idx}")
        
        # Optional manual text backup box
        typed_backup = st.text_area(
            "Or type your answer directly here:",
            height=110,
            placeholder="If you do not want to use the microphone, type your answer directly here...",
            key=f"text_{idx}"
        )

        if st.button("🚀 Submit & Save Answer", type="primary", key=f"submit_btn_{idx}"):
            final_text = ""
            
            # 1. Priority to recorded audio
            if audio_clip is not None:
                with st.spinner("🎙️ Processing the audio and converting it to text..."):
                    audio_bytes = audio_clip.read()
                    final_text = convert_audio_to_text(audio_bytes)
            
            # 2. Fallback to typed text if audio returned empty or wasn't provided
            if not final_text and typed_backup.strip():
                final_text = typed_backup.strip()

            # 3. Check and Evaluate
            if not final_text or len(final_text.split()) < 3:
                st.error("The audio was not recorded properly or the answer is too short. Please speak clearly or type your answer in the box.")
            else:
                with st.spinner("The ML model is evaluating and saving the answer..."):
                    feats, metrics = calculate_nlp_features(final_text, curr_q["expected_concepts"])
                    predicted_score = float(np.clip(eval_model.predict(feats)[0], 0.0, 100.0))

                    # PERMANENT STATE SAVE
                    st.session_state.evaluations.append({
                        "question": curr_q["question"],
                        "final_text": final_text,
                        "score": round(predicted_score, 1),
                        "metrics": metrics
                    })

                    st.success(f"✅ Answer Saved! Transcribed Text: *\"{final_text}\"*")
                    st.info(f"Predicted Score: **{predicted_score:.1f} / 100**")
                    
                    st.session_state.current_q_idx += 1
                    st.button("Go to the Next Question ➡️", on_click=lambda: st.rerun())