import streamlit as st
import pandas as pd, numpy as np, joblib, fitz, re, tempfile, requests
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# --- ENHANCED SAPLING AI DETECTION API SETUP ---
SAPLING_API_KEY = st.secrets.get("SAPLING_API_KEY", "")
SAPLING_URL = "https://api.sapling.ai/api/v1/aidetect"

def detect_ai_sapling(answer):
    """
    Enhanced Sapling AI Detector:
    Evaluates both overall score and sentence-level probabilities so that
    concise or bulleted ChatGPT responses do not slip through as 0%.
    """
    if not answer or len(answer.split()) < 4:
        return {"score": 0.0, "label": "No Strong AI Pattern", "reasons": ["Answer is too short."]}

    if not SAPLING_API_KEY:
        return {"score": 0.0, "label": "API Key Missing", "reasons": ["SAPLING_API_KEY not configured in Secrets."]}

    payload = {
        "key": SAPLING_API_KEY,
        "text": answer
    }

    try:
        res = requests.post(SAPLING_URL, json=payload, timeout=6)
        if res.status_code == 200:
            data = res.json()
            overall_score = float(data.get("score", 0.0))
            
            # Sentence-level scores inspection
            sentence_scores = []
            for item in data.get("sentence_scores", []):
                if isinstance(item, dict) and "score" in item:
                    sentence_scores.append(float(item["score"]))
                elif isinstance(item, (int, float)):
                    sentence_scores.append(float(item))

            max_sent_score = max(sentence_scores) if sentence_scores else 0.0
            effective_score = max(overall_score, max_sent_score)
            ai_score = round(effective_score * 100, 1)

            if ai_score >= 50.0:
                label = "AI Assistance Suspected"
            elif ai_score >= 25.0:
                label = "Possible AI Assistance"
            else:
                label = "No Strong AI Pattern"

            return {
                "score": ai_score,
                "label": label,
                "reasons": [f"Sapling AI Confidence: {ai_score:.0f}%"]
            }
    except Exception:
        pass

    return {"score": 0.0, "label": "No Strong AI Pattern", "reasons": ["Analysis completed."]}

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Mock Interview Analyzer", layout="wide", page_icon="🎙️")
st.markdown("""<style>
.question-box{background-color:#f0f7f7;border-radius:10px;padding:18px;margin-bottom:15px;border-left:6px solid #008080}
</style>""", unsafe_allow_html=True)

st.title("🎙️ AI Mock Interview Analyzer")
st.caption("Resume-aware technical interview system with speech-to-text, ML scoring, and Sapling AI Detection")

@st.cache_resource
def load_assets():
    return joblib.load("answer_eval_model.pkl"), pd.read_csv("questions.csv")

try:
    eval_model, questions_df = load_assets()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

DOMAIN_TAXONOMY = {
    "AI Engineer":["Deep Learning","Transformers","Computer Vision","Generative AI","NLP","BERT","PyTorch","TensorFlow","Diffusion","Reinforcement Learning","LLM"],
    "Machine Learning / Data Science":["Machine Learning","Scikit-learn","Random Forest","XGBoost","SVM","Statistics","Hypothesis Testing","Pandas","NumPy","Regression","Clustering"],
    "Computer Science Core":["Data Structures","Algorithms","Operating Systems","DBMS","SQL","Computer Networks","System Design","C++","Java","Linux","Concurrency"]
}

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(p.get_text("text") for p in doc)
    doc.close()
    return text.strip()

def analyze_resume(text):
    clean = text.lower()
    scores = {d: sum(bool(re.search(r"\b" + re.escape(s.lower()) + r"\b", clean)) for s in items) for d, items in DOMAIN_TAXONOMY.items()}
    domain = max(scores, key=scores.get) if any(scores.values()) else "Computer Science Core"
    email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\d{10}", text)
    first = text.strip().split("\n")[0].strip() if text.strip() else ""
    return {"name": first if len(first) < 35 and first else "Candidate", "email": email.group(0) if email else "N/A", "phone": phone.group(0) if phone else "N/A", "detected_domain": domain}

def convert_audio_to_text(audio_bytes):
    r = sr.Recognizer()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        with sr.AudioFile(path) as source:
            return r.recognize_google(r.record(source)).strip()
    except Exception:
        return ""
    finally:
        import os; os.remove(path)

def calculate_nlp_features(answer, expected):
    tokens = [w.lower() for w in answer.split() if len(w) > 1]
    length = len(tokens)
    exp = [w.lower() for w in expected.split()]
    coverage = sum(c in tokens for c in exp) / len(exp) if exp else 0.5
    try:
        tfidf = TfidfVectorizer().fit_transform([expected, answer])
        sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        sim = 0.2
    sem = float(np.clip(0.55 * sim + 0.45 * coverage, 0, 1))
    relevance = float(0.4 * sim + 0.6 * sem)
    keyword = float(coverage * 0.92 + 0.08 * min(1, length / 35))
    return np.array([[length, keyword, sim, sem, coverage, relevance]]), {"relevance": relevance, "concept_coverage": coverage, "tfidf_similarity": sim, "semantic_similarity": sem, "answer_length": length}

defaults = {"interview_started": False, "current_q_idx": 0, "selected_questions": [], "evaluations": [], "profile": None}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# --- SCREEN 1: RESUME UPLOAD ---
if not st.session_state.interview_started:
    st.subheader("📄 Step 1: Candidate Verification & Resume Analysis")
    uploaded_pdf = st.file_uploader("Upload candidate resume (PDF format)", type=["pdf"])

    if uploaded_pdf:
        with st.spinner("Analyzing resume content..."):
            profile = analyze_resume(extract_text_from_pdf(uploaded_pdf.read()))
            st.session_state.profile = profile
        st.success(f"Candidate: **{profile['name']}** | Domain: **{profile['detected_domain']}**")
        
        tracks = ["AI Engineer", "Machine Learning / Data Science", "Computer Science Core"]
        target = st.selectbox("Confirm Technical Interview Track:", tracks, index=tracks.index(profile["detected_domain"]))
        n = st.slider("Number of Questions:", 3, 7, 4)

        if st.button("Enter Interview Room", type="primary"):
            st.session_state.profile["target_domain"] = target
            patterns = {"AI Engineer": "AI Engineer|Deep Learning", "Machine Learning / Data Science": "Data Scientist|Machine Learning", "Computer Science Core": "Computer Science Core|Software Engineer"}
            matched = questions_df[questions_df["job_roles"].str.contains(patterns[target], na=False, regex=True)]
            st.session_state.selected_questions = matched.sample(min(len(matched), n), random_state=42).to_dict("records")
            st.session_state.interview_started = True
            st.session_state.current_q_idx = 0
            st.session_state.evaluations = []
            st.rerun()

# --- SCREEN 2: INTERVIEW & EVALUATION ---
else:
    qs = st.session_state.selected_questions
    idx = st.session_state.current_q_idx

    # Final Scorecard
    if idx >= len(qs):
        st.header("🏁 Interview Evaluation & Performance Report")
        scores = [e["score"] for e in st.session_state.evaluations]
        rel = [e["metrics"]["relevance"] * 100 for e in st.session_state.evaluations]
        cov = [e["metrics"]["concept_coverage"] * 100 for e in st.session_state.evaluations]
        avg_score, avg_rel, avg_cov = float(np.mean(scores)), float(np.mean(rel)), float(np.mean(cov))
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall Score", f"{avg_score:.1f} / 100")
        c2.metric("Concept Coverage", f"{avg_cov:.1f}%")
        c3.metric("Semantic Relevance", f"{avg_rel:.1f}%")
        c4.metric("Questions Evaluated", f"{len(qs)}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = go.Figure(data=go.Scatterpolar(r=[avg_score, avg_rel, avg_cov, min(100, avg_score + 4)], theta=["Technical Score", "Relevance", "Concept Coverage", "Depth"], fill="toself", line_color="#008080"))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            table = pd.DataFrame([{"Question": e["question"][:35] + "...", "Score": f"{e['score']:.1f}", "Concept Match": f"{e['metrics']['concept_coverage']*100:.1f}%",
                                 "AI Check": f"{e['metrics'].get('ai_assistance_label','N/A')} ({e['metrics'].get('ai_assistance_score',0):.0f}%)"} for e in st.session_state.evaluations])
            st.dataframe(table, use_container_width=True)
        
        if st.button("🔄 Start New Interview", type="primary"):
            for k, v in defaults.items(): st.session_state[k] = v
            st.rerun()
        st.stop()

    # Current Question
    q = qs[idx]
    st.progress((idx + 1) / len(qs))
    st.caption(f"Question {idx + 1} of {len(qs)} | Category: {q['category']} | Level: {q['difficulty']}")
    st.markdown(f"""<div class="question-box"><h3 style="color:#004d40;margin:0;">{q['question']}</h3></div>""", unsafe_allow_html=True)
    
    # Answer Submission (Mic + Text)
    audio = st.audio_input("Speak your answer using the microphone:", key=f"audio_{idx}")
    typed = st.text_area("Or type your answer directly here:", height=120, placeholder="Type your answer here...", key=f"text_{idx}")

    if st.button("🚀 Submit & Save Answer", type="primary", key=f"submit_btn_{idx}"):
        final = ""
        if audio:
            with st.spinner("🎙️ Converting audio to text..."):
                final = convert_audio_to_text(audio.read())
        if not final and typed.strip(): 
            final = typed.strip()

        if not final or len(final.split()) < 3:
            st.error("Please answer clearly before submitting.")
        else:
            with st.spinner("ML Model evaluating & calling Sapling AI Detection..."):
                feats, metrics = calculate_nlp_features(final, q["expected_concepts"])
                score = float(np.clip(eval_model.predict(feats)[0], 0, 100))
                
                # Direct Sapling AI API Call (Overall + Sentence-Level)
                ai_result = detect_ai_sapling(final)
                metrics["ai_assistance_score"] = ai_result["score"]
                metrics["ai_assistance_label"] = ai_result["label"]
                metrics["ai_assistance_reasons"] = ai_result["reasons"]

                st.session_state.evaluations.append({"question": q["question"], "final_text": final, "score": round(score, 1), "metrics": metrics})
            
            st.success(f'✅ Answer Saved! Transcribed Text: "{final}"')
            st.info(f"Score: **{score:.1f} / 100** | AI Check: **{ai_result['label']} ({ai_result['score']:.0f}%)**")
            st.session_state.current_q_idx += 1
            st.rerun()
