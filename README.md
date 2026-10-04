# 🎙️ AI Mock Interview Analyzer

### An End-to-End AI-Powered Technical Interview Simulation & Performance Analysis Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Dashboard-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

> Resume → Skill Analysis → Domain Detection → Mock Interview → NLP/ML Evaluation → Performance Dashboard

## 🚀 Live Demo

### [Launch AI Mock Interview Analyzer](https://ai-mock-interview-analyzer.streamlit.app/)

---

## 📌 Overview

**AI Mock Interview Analyzer** is an AI-powered technical interview simulation platform that provides candidates with a personalized mock interview experience.

The application analyzes a candidate's resume, identifies relevant technical skills and domains, selects suitable interview questions, accepts answers through **text or voice**, evaluates responses using **Natural Language Processing (NLP)** and **Machine Learning**, and presents the results through an interactive performance dashboard.

---

## ✨ Key Features

### 📄 Smart Resume Parsing
- Upload resume in PDF format
- Extract resume text using PyMuPDF
- Identify technical skills, tools, projects and experience keywords
- Use extracted information for interview personalization

### 🧭 Domain Identification
The system identifies the candidate's relevant technical domain and maps it to the appropriate interview question category.

### 🎤 Voice & Text Answers
Candidates can answer interview questions using:
- 🎤 Voice input
- ⌨️ Text input

### 🧠 NLP-Based Answer Analysis
Candidate answers are analyzed for:
- Relevance
- Technical concepts
- Keyword coverage
- Clarity
- Concept density
- Answer depth

### 🤖 Machine Learning Evaluation
A custom Scikit-Learn model evaluates answer quality.

Model:

```text
answer_eval_model.pkl
