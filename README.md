<h1 align="center">🛡️ ReviewGuard AI System</h1>
<p align="center">
  An advanced AI-powered system for Sentiment Analysis, Fake Review Detection, & Explainability using LLMs, ML Models, OCR, and LIME.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Review%20Analysis-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/NLP-BERT%2FTransformers-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-RandomForest-orange?style=for-the-badge" />
</p>

---

## 🌟 Overview

**ReviewGuard AI** is a production-grade system built to analyze e-commerce reviews from platforms like Amazon & Flipkart.  
It combines:

- 🧠 **Sentiment Classification** (BERT-based Transformer)  
- ⚠ **Fake Review Detection** (ML model + handcrafted features)  
- 🧾 **OCR Support** (Text extraction from PDF/Image/Camera)  
- 🧠 **LIME Explainability**  
- 📊 **Insights Dashboard**  
- 🎨 **Modern UI with Streamlit**  

This project demonstrates real-world review intelligence used by large e-commerce companies.

---

## ✨ Features

### 🔍 **1. Sentiment Analysis (Transformer Model)**
- Powered by **DistilBERT / BERT**  
- Fast, optimized, domain-trained  
- Supports **Text + PDF + Image + Camera**  
- Output: `Positive / Neutral / Negative`  

---

### ⚠ **2. Fake Review Detector**
- ML pipeline using:
  - TF–IDF Vectorizer  
  - RandomForest Classifier
- Trained on cleaned & balanced dataset  
- High accuracy on unseen data  

---

### 🧠 **3. Explainability (LIME)**
- Why did the model say “Fake”?  
- Highlights influential words  
- HTML-based interactive explanation  

---

### 📊 **4. Insights Dashboard**
- Sentiment distribution  
- Fake vs Genuine graph  
- WordCloud  
- Dataset insights  

---

## 🖼️ UI Highlights
- Clean modern layout  
- Choose-one input UI  
- Smooth OCR workflow  
- Animated gradients  
- Professional theme  

---

## 📁 Project Structure

ReviewGuard-AI-System/
│
├── app.py # Main Streamlit app launcher
├── pages/ # Streamlit multi-pages
│ ├── 1_Sentiment_Analysis.py
│ ├── 2_Fake_Review_Detector.py
│ ├── 3_Insights_Dashboard.py
│ └── 4_Explainability_LIME.py
│
├── app_src/
│ ├── data/processed/ # Cleaned datasets
│ ├── models/ # (Local models - excluded from GitHub)
│ ├── pipeline/ # Training scripts
│ └── utils/ # Cleaning utilities
│
├── requirements.txt
└── README.md


---

## 🧰 Tech Stack

### **AI & ML**
- BERT / DistilBERT (HuggingFace)
- PyTorch
- RandomForest
- TF–IDF
- LIME Explainability
- OCR (Tesseract)

### **Frontend**
- Streamlit (Modern UI)
- Plotly
- WordCloud

### **Backend**
- Python 3.11
- Modular pipeline scripts

---

© 2025 ReviewGuard AI — Developed by Manish Pal.
All rights reserved.

This project is licensed under the MIT License.
Redistribution allowed with proper attribution.