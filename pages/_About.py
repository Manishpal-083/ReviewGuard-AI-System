import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")

st.markdown("<h1 style='text-align:center;'>ℹ️ About ReviewGuard AI</h1>", unsafe_allow_html=True)

st.write("---")

st.subheader("👨‍💻 Developer")
st.markdown("""
### **Manish Pal**
- B.Tech AI & DS Student  
- Passionate about AI/ML, Deep Learning, and Full-Stack Development  
- Builds modern AI projects and real-world applications  
""")

st.subheader("🚀 About This Project")
st.markdown("""
ReviewGuard AI is a modern sentiment & fake-review analysis system built with:
- **DistilBERT Transformers**
- **RandomForest Classifier**
- **LIME Explainability**
- **OCR for PDF/Image/Camera**
- **Streamlit UI**
""")

st.write("---")

st.markdown("""
<div style='text-align:center; opacity:0.7'>
© 2025 ReviewGuard AI — Built with ❤️ by <b>Manish Pal</b>
</div>
""", unsafe_allow_html=True)
