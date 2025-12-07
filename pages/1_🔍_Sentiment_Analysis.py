import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import os

st.title("🔍 Sentiment Analysis")
st.write("Select an input method below to analyze sentiment.")

# ----------------------------------------
# LOAD MODEL (FALLBACK IF LOCAL MODEL MISSING)
# ----------------------------------------
@st.cache_resource
def load_model():
    local_path = "app_src/models/sentiment_model"
    weight_file = os.path.join(local_path, "pytorch_model.bin")

    if os.path.exists(weight_file):
        st.info("Loading local fine-tuned model...")
        model = DistilBertForSequenceClassification.from_pretrained(local_path)
        tokenizer = DistilBertTokenizerFast.from_pretrained(local_path)
    else:
        st.warning("Local model weights missing. Using pretrained SST-2 model instead.")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
    return model, tokenizer

model, tokenizer = load_model()

# ----------------------------------------
# OCR FUNCTIONS
# ----------------------------------------
def extract_from_pdf(pdf):
    pages = convert_from_bytes(pdf.read())
    text = ""
    for pg in pages:
        text += pytesseract.image_to_string(pg)
    return text

def extract_from_image(img):
    return pytesseract.image_to_string(Image.open(img))

# ----------------------------------------
# INPUT TYPE
# ----------------------------------------
choice = st.radio(
    "Choose Input Type:",
    ["✏ Text", "📄 PDF / 🖼 Image", "📷 Camera"],
    horizontal=True
)

final_text = ""

# ----------------------------------------
# TEXT INPUT
# ----------------------------------------
if choice == "✏ Text":
    st.subheader("📝 Enter Text")
    final_text = st.text_area("Write your review here:", height=130)

# ----------------------------------------
# PDF / IMAGE
# ----------------------------------------
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

# ----------------------------------------
# CAMERA INPUT
# ----------------------------------------
elif choice == "📷 Camera":
    st.subheader("Capture using Camera")
    cam_img = st.camera_input("Take a picture")

    if cam_img:
        st.info("Extracting text from captured image...")
        final_text = extract_from_image(cam_img)
        st.success("Text extracted successfully!")

# ----------------------------------------
# SENTIMENT PREDICTION
# ----------------------------------------
if st.button("Analyze Sentiment"):
    if not final_text.strip():
        st.error("⚠ No input found. Please provide some text.")
    else:
        encoded = tokenizer(final_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            prediction = model(**encoded)
            pred = torch.argmax(prediction.logits).item()

        labels = ["Negative 😡", "Positive 😄"]  # SST-2 only has 2 labels
        st.success(f"### Sentiment → **{labels[pred]}**")
