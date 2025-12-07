import streamlit as st
from PIL import Image
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
import torch
import fitz  # PyMuPDF for PDF text extraction

st.title("🔍 Sentiment Analysis ")
st.write("Analyze sentiment from text, PDF, image, or camera input.")

# -----------------------------
# LOAD SENTIMENT MODEL
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    try:
        # Try loading local model
        model = DistilBertForSequenceClassification.from_pretrained(
            "app_src/models/sentiment_model"
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            "app_src/models/sentiment_model"
        )
    except:
        st.warning("Local model not found. Using pretrained SST-2 model.")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
    return model, tokenizer


model, tokenizer = load_sentiment_model()

# -----------------------------
# LOAD TROCŔ OCR MODEL
# -----------------------------
@st.cache_resource
def load_ocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    return processor, ocr_model


processor, ocr_model = load_ocr_model()


# -----------------------------
# OCR FUNCTIONS
# -----------------------------
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyMuPDF (Cloud compatible)."""
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_with_trocr(image):
    """Extract text from images using HuggingFace TrOCR."""
    img = Image.open(image).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


# -----------------------------
# INPUT SELECTION
# -----------------------------
choice = st.radio(
    "Choose Input Type:",
    ["✏ Text", "📄 PDF / 🖼 Image", "📷 Camera"],
    horizontal=True
)

final_text = ""


# -----------------------------
# TEXT INPUT
# -----------------------------
if choice == "✏ Text":
    final_text = st.text_area("Enter text:", height=130)


# -----------------------------
# PDF / IMAGE INPUT
# -----------------------------
elif choice == "📄 PDF / 🖼 Image":
    file = st.file_uploader("Upload PDF, PNG, JPG", type=["pdf", "png", "jpg", "jpeg"])

    if file:
        if file.type == "application/pdf":
            st.info("Extracting text from PDF...")
            final_text = extract_text_from_pdf(file)
        else:
            st.info("Extracting text from Image using AI OCR...")
            final_text = extract_text_with_trocr(file)

        st.success("Text extracted successfully!")


# -----------------------------
# CAMERA INPUT
# -----------------------------
elif choice == "📷 Camera":
    cam_img = st.camera_input("Take a picture")

    if cam_img:
        st.info("Extracting text from camera image...")
        final_text = extract_text_with_trocr(cam_img)
        st.success("Text extracted successfully!")


# -----------------------------
# SENTIMENT PREDICTION
# -----------------------------
if st.button("Analyze Sentiment"):
    if not final_text.strip():
        st.error("⚠ No input found. Please provide something.")
    else:
        encoded = tokenizer(final_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            pred = model(**encoded).logits.argmax().item()

        labels = ["Negative 😡", "Positive 😄"]
        st.success(f"### Sentiment → **{labels[pred]}**")
