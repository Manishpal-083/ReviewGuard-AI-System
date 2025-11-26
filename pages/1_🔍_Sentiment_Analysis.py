import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

st.title("🔍 Sentiment Analysis")
st.write("Select an input method below to analyze sentiment.")

# -----------------------
# LOAD MODEL (DEPLOY SAFE)
# -----------------------
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("app_src/models/sentiment_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("app_src/models/sentiment_model")
    return model, tokenizer

model, tokenizer = load_model()

# -----------------------
# OCR FUNCTIONS
# -----------------------
def extract_from_pdf(pdf):
    pages = convert_from_bytes(pdf.read())
    text = ""
    for pg in pages:
        text += pytesseract.image_to_string(pg)
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
    final_text = st.text_area("Write your review here:", height=130)


# -----------------------
# 2️⃣ PDF / IMAGE INPUT
# -----------------------
elif choice == "📄 PDF / 🖼 Image":
    st.subheader("Upload PDF or Image")

    file = st.file_uploader("Upload PDF, PNG, JPG", type=["pdf", "png", "jpg", "jpeg"])

    if file:
        if file.type == "application/pdf":
            st.info("Extracting text from PDF...")
            final_text = extract_from_pdf(file)
        else:
            st.info("Extracting text from Image...")
            final_text = extract_from_image(file)

        st.success("Text extracted successfully!")


# -----------------------
# 3️⃣ CAMERA INPUT
# -----------------------
elif choice == "📷 Camera":
    st.subheader("Capture using Camera")

    cam_img = st.camera_input("Take a picture")
    if cam_img:
        st.info("Extracting text from captured image...")
        final_text = extract_from_image(cam_img)
        st.success("Text extracted successfully!")


# -----------------------
# PREDICT BUTTON
# -----------------------
if st.button("Analyze Sentiment"):
    if not final_text.strip():
        st.error("⚠ No input found. Please provide text, PDF, image, or camera input.")
    else:
        encoded = tokenizer(final_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            prediction = model(**encoded)
            pred = torch.argmax(prediction.logits).item()

        labels = ["Negative 😡", "Neutral 😐", "Positive 😄"]
        st.success(f"### Sentiment → **{labels[pred]}**")
