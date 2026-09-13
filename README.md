AI-Mock-Interview-Analyzer

An AI-powered mock interview system that analyzes resumes, conducts technical interviews, evaluates candidate answers using Machine Learning and NLP, and generates performance insights.

Features

Resume analysis and technical skill detection

Automatic technical-domain identification

Customized interview track selection

Voice-based answer submission

Text-based answer submission

NLP-based answer relevance and concept analysis

Machine Learning-based answer scoring

Question-wise evaluation

Final performance report with visualization

Technologies

Python, Streamlit, Machine Learning, NLP, Scikit-learn, Pandas, NumPy, PyMuPDF, SpeechRecognition, Plotly, and Joblib.

Run Locally

pip install -r requirements.txt
streamlit run app.py

Project Files

app.py — Streamlit application

answer_eval_model.pkl — trained answer evaluation model

questions.csv — technical interview question dataset

MODEL.ipynb — model development/training notebook

answer_evaluation.csv — evaluation/training data