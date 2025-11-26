import streamlit as st
from lime.lime_text import LimeTextExplainer
import joblib
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import streamlit.components.v1 as components

st.title("🧠 LIME Explainability")
st.write("Understand why the model predicted Fake/Genuine.")

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    model = joblib.load("app_src/models/fake_review_model.pkl")
    vec = joblib.load("app_src/models/fake_vectorizer.pkl")
    return model, vec

model, vec = load_model()

# OCR FUNCTIONS
def extract_from_pdf(pdf):
    pages = convert_from_bytes(pdf.read())
    return "".join([pytesseract.image_to_string(p) for p in pages])

def extract_from_image(img):
    return pytesseract.image_to_string(Image.open(img))

# INPUT TYPE SELECTION
choice = st.radio(
    "Choose Input Type:",
    ["✏ Text", "📄 PDF / 🖼 Image", "📷 Camera"],
    horizontal=True,
)

final_text = ""

# 1) TEXT
if choice == "✏ Text":
    final_text = st.text_area("Enter review text")

# 2) PDF/IMAGE
elif choice == "📄 PDF / 🖼 Image":
    file = st.file_uploader("Upload File", type=["pdf","png","jpg","jpeg"])
    if file:
        if file.type == "application/pdf":
            final_text = extract_from_pdf(file)
        else:
            final_text = extract_from_image(file)
        st.success("Text extracted!")

# 3) CAMERA
elif choice == "📷 Camera":
    cam = st.camera_input("Take a picture")
    if cam:
        final_text = extract_from_image(cam)
        st.success("Text extracted!")


# EXPLAIN BUTTON
if st.button("Explain"):
    if not final_text.strip():
        st.error("Enter some text first")
    else:
        explainer = LimeTextExplainer(class_names=["Genuine","Fake"])

        explanation = explainer.explain_instance(
            final_text,
            lambda x: model.predict_proba(vec.transform(x)),
            num_features=10
        )

        st.subheader("🔎 Key Influential Words")
        for w, score in explanation.as_list():
            st.write(f"**{w} → {score:.3f}**")

        st.write("### LIME Visual Explanation")
        components.html(explanation.as_html(), height=600, scrolling=True)
