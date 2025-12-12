<h1 align="center">🛡️ ReviewGuard AI System</h1>
<p align="center">
  An advanced AI-powered system for Sentiment Analysis, Fake Review Detection & Explainability using LLMs, ML Models, OCR, and LIME.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Review%20Analysis-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/NLP-BERT%2FTransformers-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-RandomForest-orange?style=for-the-badge" />
</p>

---

## 🌟 Overview

**ReviewGuard AI** is a production-grade system for analyzing e-commerce reviews (Amazon, Flipkart, Google Reviews).  
It integrates:

- 🧠 **BERT-based Sentiment Analysis**  
- ⚠ **Fake Review Detection using ML**  
- 🧾 **OCR for PDF/Image/Camera reviews**  
- 🧠 **LIME Explainability for transparency**  
- 📊 **Interactive Insights Dashboard**  
- 🎨 **Modern Streamlit UI with animations**  

Built to replicate real-world review intelligence pipelines used in large-scale e-commerce platforms.

---

## ✨ Features

### 🔍 **1. Sentiment Analysis (Transformer Model)**
- Powered by **DistilBERT / BERT**  
- Domain-optimized & fast  
- Works on:  
  - Plain Text  
  - PDF  
  - Image  
  - Camera input  
- Output: `Positive | Neutral | Negative`

---

### ⚠ **2. Fake Review Detector**
- ML pipeline with:
  - TF–IDF Vectorizer  
  - RandomForest Classifier  
- Balanced dataset training  
- Robust against noisy reviews  
- High generalization on unseen data  

---

### 🧠 **3. Explainability (LIME)**
- Highlights the specific words that influenced prediction  
- Generates interactive HTML explanation  
- Helps users trust AI decisions  

---

### 📊 **4. Insights Dashboard**
Includes visual insights such as:

- Sentiment distribution  
- Fake vs Genuine comparison  
- Word clouds  
- Category-wise insights  
- Review patterns  

---

## 🖼️ UI Highlights

- Clean modern layout  
- Glassmorphism theme  
- Smooth OCR workflow  
- Streamlit multipage navigation  
- Animated progress & transitions  

---

## 📁 Project Structure

ReviewGuard-AI-System/
│
├── app.py # Main Streamlit launcher
│
├── pages/ # Streamlit multi-page UI
│ ├── 1_Sentiment_Analysis.py
│ ├── 2_Fake_Review_Detector.py
│ ├── 3_Insights_Dashboard.py
│ └── 4_Explainability_LIME.py
│
├── app_src/
│ ├── data/processed/ # Cleaned datasets
│ ├── models/ # Local models (excluded from GitHub)
│ ├── pipeline/ # Training scripts
│ └── utils/ # Utility modules (OCR, helpers)
│
├── requirements.txt
└── README.md


## 🧰 Tech Stack

### **AI & NLP**
- BERT / DistilBERT (HuggingFace Transformers)  
- PyTorch  
- LIME Explainability  
- TF–IDF Vectorization  
- RandomForest Classifier  

### **OCR**
- Tesseract OCR  
- PyMuPDF (`fitz`)  
- pdfplumber  

### **Frontend**
- Streamlit (Modern UI)  
- Plotly  
- WordCloud  

### **Backend**
- Python 3.11  
- Modular training & inference pipeline  

---

## 🚀 How It Works (High-Level)

1. User inputs review text **or** uploads PDF/Image  
2. OCR extracts text (if needed)  
3. Text goes through:  
   - Preprocessing  
   - Transformer sentiment model  
   - Fake review classifier  
4. LIME generates explanation  
5. Dashboard visualizes insights  

---

## 🛠 Future Improvements

- Replace classical ML classifier with **LLM-based Fake Review Detection**  
- Add radar charts for review scoring  
- Integrate AWS / GCP for scalable inference  
- Build ReviewGuard Chrome Extension  
- Multi-language sentiment support  

---

## © Copyright

© 2025 **ReViewGuard AI — Developed by Manish Pal**  
All rights reserved.  

This project is licensed under the **MIT License**.  
Redistribution allowed with proper attribution.
