import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import joblib

st.title("⚠ Fake Review Detector")
st.write("Choose an input method to detect if a review is Fake or Genuine.")

# -----------------------
# LOAD MODEL (DEPLOY SAFE)
# -----------------------
@st.cache_resource
def load_model():
    model = joblib.load("app_src/models/fake_review_model.pkl")
    vectorizer = joblib.load("app_src/models/fake_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()


# -----------------------
# OCR FUNCTIONS
# -----------------------
def extract_from_pdf(pdf):
    pages = convert_from_bytes(pdf.read())
    text = ""
    for p in pages:
        text += pytesseract.image_to_string(p)
    return text

def extract_from_image(img):
    return pytesseract.image_to_string(Image.open(img))


# -----------------------
# INPUT SELECTION
# -----------------------
choice = st.radio(
    "Choose Input Type:",
    ["✏ Text", "📄 PDF / 🖼 Image", "📷 Camera"],
    horizontal=True
)

final_text = ""

# -----------------------
# 1️⃣ TEXT INPUT
# -----------------------
if choice == "✏ Text":
    st.subheader("📝 Enter Text")
    final_text = st.text_area("Write your review:", height=130)

# -----------------------
# 2️⃣ PDF / IMG INPUT
# -----------------------
elif choice == "📄 PDF / 🖼 Image":
    st.subheader("Upload PDF or Image")
    file = st.file_uploader("Upload PDF / Image", type=["pdf", "png", "jpg", "jpeg"])

    if file:
        if file.type == "application/pdf":
            st.info("Extracting text from PDF…")
            final_text = extract_from_pdf(file)
        else:
            st.info("Extracting text from Image…")
            final_text = extract_from_image(file)

        st.success("Text extracted successfully!")

# -----------------------
# 3️⃣ CAMERA INPUT
# -----------------------
elif choice == "📷 Camera":
    st.subheader("Capture using Camera")
    cam_file = st.camera_input("Capture Image")
    if cam_file:
        st.info("Extracting text from camera capture…")
        final_text = extract_from_image(cam_file)
        st.success("Text extracted successfully!")

# -----------------------
# PREDICT BUTTON
# -----------------------
if st.button("Check Review"):
    if not final_text.strip():
        st.error("⚠ Please provide some input first!")
    else:
        vec = vectorizer.transform([final_text])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.error("⚠ This review looks **FAKE**")
        else:
            st.success("✅ This review appears **GENUINE**")
