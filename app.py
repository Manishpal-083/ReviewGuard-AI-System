
import streamlit as st
import streamlit.components.v1 as components
import time

st.set_page_config(
    page_title="ReviewGuard AI",
    page_icon="🛡",
    layout="wide",
)

# GLOBAL CSS -----------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Glass Card */
.glass-card {
    background: rgba(255,255,255,0.08);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.2);
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

/* Footer */
.footer {
    text-align:center;
    padding: 15px;
    margin-top:40px;
    opacity:0.6;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# HEADER --------------------------------------------------------
st.markdown("<h1 style='text-align:center; font-size:60px; font-weight:900; background:linear-gradient(90deg,#a855f7,#6366f1,#ec4899); -webkit-text-fill-color:transparent; -webkit-background-clip:text;'>ReviewGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#cbd5e1; font-size:20px;'>Next-Gen AI Product Review Analyzer</p>", unsafe_allow_html=True)

# FEATURE CARDS ------------------------------------------------
st.markdown("""
<div class="glass-card">
<h3>✨ Features</h3>
<ul style='font-size:18px;line-height:1.8'>
<li>🔍 AI Sentiment Analysis (DistilBERT)</li>
<li>⚠ Fake Review Detection (RandomForest)</li>
<li>📊 Insightful Dashboard Visuals</li>
<li>🧠 Explainability using LIME</li>
<li>📄 OCR for PDF, Images & Camera Input</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.info("⭐ Use the left sidebar to explore all features")

# FOOTER --------------------------------------------------------
st.markdown("""
<div class="footer">
    © 2025 ReviewGuard AI · Built with ❤️ by Manish Pal
</div>
""", unsafe_allow_html=True)
