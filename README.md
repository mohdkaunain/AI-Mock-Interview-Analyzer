<div align="center">

# 🎙️ AI Mock Interview Analyzer

**Upload a resume, get interviewed on your own skills, and see exactly how you scored.**

Resume → Skill Analysis → Domain Detection → Mock Interview → NLP/ML Evaluation → Performance Dashboard

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ai-mock-interview-analyzer.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

### 🚀 [Try the live demo](https://ai-mock-interview-analyzer.streamlit.app/)

</div>

---

## 📌 Overview

AI Mock Interview Analyzer gives you a personalized technical interview without needing another person on the call.

It reads your resume, works out which technical domain you belong to, picks questions for that domain, and takes your answers by text or voice. Each answer is scored with NLP checks and a trained ML model, and at the end you get a dashboard showing your strengths and gaps.

## ✨ Key Features

### 📄 Smart resume parsing
- Upload your resume as a PDF
- Text is extracted with PyMuPDF
- Skills, tools, projects and experience keywords are picked out
- That context is used to personalize the interview

### 🧭 Domain identification
Extracted skills are mapped to a domain, and the matching question category is loaded from `questions.csv`.

### 🎤 Voice and text answers
Answer however you're comfortable:
- 🎤 Voice, using SpeechRecognition
- ⌨️ Text, typed straight into the app

### 🧠 NLP answer analysis
Every answer is checked for:
- Relevance to the question
- Technical concepts used
- Keyword coverage
- Clarity
- Concept density
- Depth

### 🤖 ML scoring
A custom Scikit-Learn model, saved as `answer_eval_model.pkl`, rates answer quality. It was trained on the labelled data in `answer_evaluation.csv`.

### 🕵️ AI-generated answer checks
Looks at structure, factual precision and depth against benchmark technical answers, so a polished but hollow answer doesn't score well.

### 🔐 API key protection
Keys are read from environment variables, sanitized, and kept out of logs and request payloads.

### 📈 Performance dashboard
An interactive Plotly radar chart plus a score breakdown for every question.

---

## 🏗️ How it works

```text
Resume (PDF)
    |
    v
PyMuPDF parser --> skills, keywords, context
    |
    v
Domain detection <-- questions.csv
    |
    v
Interview (Streamlit, app.py)
    |-- 🎤 voice input (SpeechRecognition)
    |-- ⌨️ text input
    v
Evaluation
    |-- NLP analysis
    |-- AI-answer detection
    |-- ML scoring (answer_eval_model.pkl)
    v
Dashboard (Plotly radar + breakdown)
```

## 🧰 Tech stack

| Area | Tools |
|------|-------|
| Language | Python |
| App / UI | Streamlit |
| ML | Scikit-learn, Joblib |
| Data | Pandas, NumPy |
| NLP | NLTK, spaCy |
| Resume parsing | PyMuPDF |
| Speech input | SpeechRecognition |
| Charts | Plotly |
| Integrations | REST API wrappers |

## 📁 Project structure

```text
AI-Mock-Interview-Analyzer/
├── app.py                  Streamlit app and main logic
├── MODEL.ipynb             EDA, feature engineering, training
├── answer_eval_model.pkl   trained scoring model
├── questions.csv           question bank by domain
├── answer_evaluation.csv   training and benchmark data
├── requirements.txt
└── README.md
```

---

## ⚙️ Run it locally

You need Python 3.9 or newer, plus a microphone if you want voice answers. On some machines SpeechRecognition's mic input also needs PyAudio, which depends on PortAudio.

**1. Clone the repo**

```bash
git clone https://github.com/<your-username>/AI-Mock-Interview-Analyzer.git
cd AI-Mock-Interview-Analyzer
```

**2. Make a virtual environment**

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Download NLP resources** (only what your code uses)

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**5. Set your API key**

Create a `.env` file in the project root and keep it out of git:

```env
API_KEY=your_api_key_here
```

```bash
echo ".env" >> .gitignore
```

**6. Start the app**

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

## 🎯 Using the app

1. Upload your resume as a PDF.
2. Check the skills and domain it detected.
3. Start the interview and answer each question by voice or text.
4. See the score for each answer.
5. Open the dashboard for the radar chart and full breakdown.

---

## 🧪 Training the model

Everything about the scoring model lives in `MODEL.ipynb`.

```bash
pip install jupyter
jupyter notebook MODEL.ipynb
```

The notebook goes through these steps:

1. **Explore** `answer_evaluation.csv`: score distribution, class balance, missing values, answer lengths.
2. **Build features** from the answers, like relevance, concept density and clarity signals.
3. **Train** the Scikit-Learn pipeline and tune it with cross-validation.
4. **Evaluate** on a held-out split before replacing the current model.
5. **Export** the model:

```python
import joblib
joblib.dump(model, "answer_eval_model.pkl")
```

`app.py` loads it with `joblib.load("answer_eval_model.pkl")`, so swapping the file and restarting Streamlit is all it takes.

Things to keep in mind when retraining:

- The feature code in the notebook must match what `app.py` does at inference time, otherwise scores will be off.
- To improve the model, add more labelled rows to `answer_evaluation.csv` and retrain.
- Keep old `.pkl` files (like `answer_eval_model_v1.pkl`) so you can roll back.
- New domains or questions go into `questions.csv`.

## 🔐 Security

- Secrets are loaded from environment variables only.
- API keys are sanitized and masked before logging or building request payloads.
- Never commit `.env` or any credentials.
- Resumes are parsed locally. If you deploy this with real candidate data, check what gets sent to any external API first.

## 🤝 Contributing

Fork the repo, make a branch, commit, push, and open a pull request.

## 📄 License

MIT. See [LICENSE](LICENSE).

---

<div align="center">

Built by Mohd Kaunain Abidi

If this helped your interview prep, a ⭐ on the repo is appreciated.

</div>
