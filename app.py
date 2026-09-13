import streamlit as st
import streamlit.components.v1 as components
import pandas as pd, numpy as np, joblib, fitz, re, tempfile
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

st.set_page_config(page_title="AI Mock Interview Analyzer", layout="wide", page_icon="🎥")

st.markdown("""<style>
.question-box{background-color:#f0f7f7;border-radius:10px;padding:18px;margin-bottom:15px;border-left:6px solid #008080}
.status-badge{display:inline-block;padding:4px 10px;border-radius:15px;background-color:#e8f5e9;color:#2e7d32;font-weight:bold;font-size:13px}
</style>""", unsafe_allow_html=True)

st.title("🎥 AI Mock Interview Analyzer")
st.caption("Resume-aware technical interview system with speech-to-text, NLP analysis, and ML-based scoring")

@st.cache_resource
def load_assets():
    return joblib.load("answer_eval_model.pkl"), pd.read_csv("questions.csv")

try:
    eval_model, questions_df = load_assets()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.info("Make sure answer_eval_model.pkl and questions.csv are present in the project directory.")
    st.stop()

DOMAIN_TAXONOMY = {
    "AI Engineer":["Deep Learning","Transformers","Computer Vision","Generative AI","NLP","BERT","PyTorch","TensorFlow","Diffusion","Reinforcement Learning","LLM"],
    "Machine Learning / Data Science":["Machine Learning","Scikit-learn","Random Forest","XGBoost","SVM","Statistics","Hypothesis Testing","Pandas","NumPy","Regression","Clustering"],
    "Computer Science Core":["Data Structures","Algorithms","Operating Systems","DBMS","SQL","Computer Networks","System Design","C++","Java","Linux","Concurrency"]
}

def extract_text_from_pdf(pdf_bytes):
    doc=fitz.open(stream=pdf_bytes,filetype="pdf")
    text="\n".join(p.get_text("text") for p in doc)
    doc.close()
    return text.strip()

def analyze_resume(text):
    clean=text.lower()
    scores={d:0 for d in DOMAIN_TAXONOMY}
    skills={}
    for domain,items in DOMAIN_TAXONOMY.items():
        found=[s for s in items if re.search(r"\b"+re.escape(s.lower())+r"\b",clean)]
        skills[domain]=found
        scores[domain]=len(found)
    domain=max(scores,key=scores.get)
    if scores[domain]==0: domain="Computer Science Core"
    email=re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",text)
    phone=re.search(r"(?:\+?\d{1,3}[-.\s]?)?\d{10}",text)
    first=text.strip().split("\n")[0].strip() if text.strip() else ""
    return {"name":first if len(first)<35 and first else "Candidate","email":email.group(0) if email else "N/A","phone":phone.group(0) if phone else "N/A","detected_domain":domain,"matched_skills":skills[domain]}

def convert_audio_to_text(audio_bytes):
    r=sr.Recognizer(); path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f:
            f.write(audio_bytes); path=f.name
        with sr.AudioFile(path) as source: data=r.record(source)
        return r.recognize_google(data).strip()
    except Exception:
        return ""
    finally:
        if path:
            try:
                import os; os.remove(path)
            except OSError: pass


def ai_assistance_check(answer):
    """Heuristic indicator only; it does not prove AI authorship."""
    words = re.findall(r"\b\w+\b", answer.lower())
    n = len(words)
    if n < 12:
        return 0.0, "Insufficient text"

    formal = [
        "in conclusion", "furthermore", "moreover", "therefore",
        "consequently", "in summary", "it is important to note",
        "plays a crucial role", "overall"
    ]
    structure = [
        "firstly", "secondly", "thirdly", "however", "for example",
        "in contrast", "on the other hand"
    ]
    formal_hits = sum(p in answer.lower() for p in formal)
    structure_hits = sum(p in answer.lower() for p in structure)
    punctuation = sum(answer.count(x) for x in [",", ";", ":"])

    long_ratio = sum(len(w) >= 9 for w in words) / n
    score = (
        0.30 * min(formal_hits / 2, 1) +
        0.20 * min(structure_hits / 2, 1) +
        0.20 * min(punctuation / 8, 1) +
        0.30 * min(long_ratio / 0.22, 1)
    )
    score = float(np.clip(score, 0, 1))

    if score >= 0.72:
        label = "AI Assistance Suspected"
    elif score >= 0.50:
        label = "Possible AI Assistance"
    else:
        label = "No Strong AI Pattern"

    return score, label

def calculate_nlp_features(answer, expected):
    tokens=[w.lower() for w in answer.split() if len(w)>1]
    length=len(tokens)
    exp=[w.lower() for w in expected.split()]
    coverage=sum(c in tokens for c in exp)/len(exp) if exp else .5
    try:
        tfidf=TfidfVectorizer().fit_transform([expected,answer])
        sim=float(cosine_similarity(tfidf[0:1],tfidf[1:2])[0][0])
    except Exception: sim=.2
    sem=float(np.clip(.55*sim+.45*coverage,0,1))
    relevance=float(.4*sim+.6*sem)
    keyword=float(coverage*.92+.08*min(1,length/35))
    return np.array([[length,keyword,sim,sem,coverage,relevance]]),{"relevance":relevance,"concept_coverage":coverage,"tfidf_similarity":sim,"semantic_similarity":sem,"answer_length":length}

defaults={"interview_started":False,"current_q_idx":0,"selected_questions":[],"evaluations":[],"profile":None}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v
if "tab_switches" not in st.session_state:
    st.session_state.tab_switches = 0


if not st.session_state.interview_started:
    st.subheader("📄 Step 1: Candidate Verification & Resume Analysis")
    uploaded_pdf=st.file_uploader("Upload candidate resume (PDF format)",type=["pdf"])

    if uploaded_pdf:
        with st.spinner("Analyzing resume content..."):
            profile=analyze_resume(extract_text_from_pdf(uploaded_pdf.read()))
            st.session_state.profile=profile
        st.success(f"Candidate Profile: **{profile['name']}** | Domain: **{profile['detected_domain']}**")
        c1,c2=st.columns(2)
        with c1:
            st.write(f"**Email:** {profile['email']}")
            st.write(f"**Phone:** {profile['phone']}")
        with c2:
            st.write(f"**Domain:** {profile['detected_domain']}")
            st.write(f"**Matched Skills:** {', '.join(profile['matched_skills']) if profile['matched_skills'] else 'General Concepts'}")
        st.divider()
        st.subheader("⚙️ Interview Track Selection")
        tracks=["AI Engineer","Machine Learning / Data Science","Computer Science Core"]
        target=st.selectbox("Confirm Technical Interview Track:",tracks,index=tracks.index(profile["detected_domain"]))
        n=st.slider("Number of Questions:",3,7,4)

        if st.button("Enter Interview Room",type="primary"):
            st.session_state.profile["target_domain"]=target
            patterns={
                "AI Engineer":"AI Engineer|Deep Learning",
                "Machine Learning / Data Science":"Data Scientist|Machine Learning",
                "Computer Science Core":"Computer Science Core|Software Engineer"
            }
            matched=questions_df[questions_df["job_roles"].str.contains(patterns[target],na=False,regex=True)]
            if matched.empty:
                st.error("No questions were found for the selected interview track."); st.stop()
            st.session_state.selected_questions=matched.sample(min(len(matched),n),random_state=42).to_dict("records")
            st.session_state.interview_started=True
            st.session_state.current_q_idx=0
            st.session_state.evaluations=[]
            st.session_state.tab_switches=0
            st.rerun()

else:
    # Browser-side focus/visibility monitor. It can warn that the interview
    # page was left, but cannot identify which other website was opened.
    st.markdown("""
    <script>
    (function() {
      if (window.__tabWarnInstalled) return;
      window.__tabWarnInstalled = true;
      document.addEventListener("visibilitychange", function() {
        if (document.visibilityState === "hidden") {
          console.log("Interview tab left");
        }
      });
    })();
    </script>
    """, unsafe_allow_html=True)

    qs=st.session_state.selected_questions
    idx=st.session_state.current_q_idx

    if idx>=len(qs):
        st.header("🏁 Interview Evaluation & Performance Report")
        profile=st.session_state.profile
        st.subheader(f"Candidate: {profile['name']} | Domain Track: {profile['target_domain']}")
        scores=[e["score"] for e in st.session_state.evaluations]
        rel=[e["metrics"]["relevance"]*100 for e in st.session_state.evaluations]
        cov=[e["metrics"]["concept_coverage"]*100 for e in st.session_state.evaluations]
        avg_score=float(np.mean(scores)) if scores else 0
        avg_rel=float(np.mean(rel)) if rel else 0
        avg_cov=float(np.mean(cov)) if cov else 0
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Overall Score",f"{avg_score:.1f} / 100")
        c2.metric("Concept Coverage",f"{avg_cov:.1f}%")
        c3.metric("Semantic Relevance",f"{avg_rel:.1f}%")
        c4.metric("Questions Evaluated",f"{len(qs)}")
        col1,col2=st.columns([1,1])
        with col1:
            st.subheader("Performance Radar Breakdown")
            fig=go.Figure(data=go.Scatterpolar(r=[avg_score,avg_rel,avg_cov,min(100,avg_score+4)],theta=["Technical Score","Relevance","Concept Coverage","Depth"],fill="toself",line_color="#008080"))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        with col2:
            st.subheader("Question-Wise Score Audit")
            table=pd.DataFrame([{"Question":e["question"][:40]+"...","Candidate Answer":e.get("final_text","")[:40]+"...","Score":f"{e['score']:.1f}","Concept Match":f"{e['metrics']['concept_coverage']*100:.1f}%",
                     "AI Check":e["metrics"].get("ai_assistance_label","N/A")} for e in st.session_state.evaluations])
            st.dataframe(table,use_container_width=True)
        if st.button("🔄 Start New Interview",type="primary"):
            for k,v in defaults.items(): st.session_state[k]=v
            st.session_state.tab_switches=0
            st.rerun()
        st.stop()

    st.caption("🛡️ Interview monitoring: leaving this page may be detectable by the browser. The app cannot identify whether another tab is ChatGPT, Google, or another site.")
    q=qs[idx]
    st.progress((idx+1)/len(qs))
    st.caption(f"Question {idx+1} of {len(qs)} | Category: {q['category']} | Level: {q['difficulty']}")
    st.markdown(f"""<div class="question-box"><h3 style="color:#004d40;margin:0;">{q['question']}</h3></div>""",unsafe_allow_html=True)
    st.subheader("🎙️ Answer Submission")
    audio=st.audio_input("Speak your answer using the microphone:",key=f"audio_{idx}")
    typed=st.text_area("Or type your answer directly here:",height=130,placeholder="If you do not want to use the microphone, type your answer directly here...",key=f"text_{idx}")

    if st.button("🚀 Submit & Save Answer",type="primary",key=f"submit_btn_{idx}"):
        final=""
        if audio:
            with st.spinner("🎙️ Processing the audio and converting it to text..."):
                final=convert_audio_to_text(audio.read())
        if not final and typed.strip(): final=typed.strip()
        if not final or len(final.split())<3:
            st.error("The audio was not recorded properly or the answer is too short. Please speak clearly or type your answer in the box.")
        else:
            with st.spinner("The ML model is evaluating and saving the answer..."):
                feats,metrics=calculate_nlp_features(final,q["expected_concepts"])
                score=float(np.clip(eval_model.predict(feats)[0],0,100))
                ai_score, ai_label = ai_assistance_check(final)
                metrics["ai_assistance_score"] = ai_score
                metrics["ai_assistance_label"] = ai_label
                st.session_state.evaluations.append({"question":q["question"],"final_text":final,"score":round(score,1),"metrics":metrics})
            st.success(f'✅ Answer Saved! Transcribed Text: "{final}"')
            st.info(f"Predicted Score: **{score:.1f} / 100**")
            if ai_label == "AI Assistance Suspected":
                st.warning(f"⚠️ **{ai_label}** — This is a heuristic indicator, not proof that AI was used.")
            elif ai_label == "Possible AI Assistance":
                st.warning(f"⚠️ **{ai_label}** — Review the answer if needed.")
            else:
                st.success("✅ No strong AI-writing pattern detected.")
            st.session_state.current_q_idx+=1
            st.rerun()
