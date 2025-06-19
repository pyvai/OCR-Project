import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from fpdf import FPDF
import io

# Update path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------------- Preprocessing -----------------------
def remove_blue_lines(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = image.copy()
    result[mask != 0] = [255, 255, 255]
    return result

def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    blur = cv2.GaussianBlur(denoised, (5, 5), 0)
    sharpen = cv2.addWeighted(denoised, 1.5, blur, -0.5, 0)
    _, binary = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def preprocess_handwritten(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    denoised = cv2.fastNlMeansDenoising(norm, None, 30, 7, 21)
    adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 21, 12)
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    processed = cv2.dilate(processed, kernel, iterations=1)
    processed = cv2.erode(processed, kernel, iterations=1)
    return cv2.bitwise_not(processed)

def preprocess_vehicle_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

# ----------------------- OCR Extraction -----------------------
def extract_text(img, mode="both"):
    if mode == "text only":
        config = "--oem 3 --psm 6 -l eng"
    elif mode == "digits only":
        config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
    elif mode == "handwritten text":
        config = "--oem 1 --psm 11 -l eng"
    elif mode == "alphabets + symbols":
        config = "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@#$%&*(){}[]+-=_!?:;.,/"
    elif mode == "vehicle number plate":
        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    else:  # both
        config = "--oem 3 --psm 6 -l eng+osd"
    return pytesseract.image_to_string(img, config=config)

# ----------------------- PDF Export -----------------------
def generate_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    buffer = io.BytesIO()
    pdf.output(buffer, 'F')
    return buffer.getvalue()

# ----------------------- Streamlit UI -----------------------
st.set_page_config("📄 OCR: Text, Digits, Symbols, Handwriting", layout="centered")
st.title("📄 Smart OCR Model for Education Sector")
st.markdown("💡 This OCR extracts handwritten notes, digits, symbols, and text from various fonts and styles.")

uploaded = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])

mode = st.selectbox("🔍 Choose OCR Mode", [
    "both (recommended)",
    "text only",
    "digits only",
    "alphabets + symbols",
    "handwritten text",
    "vehicle number plate"
])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="🖼️ Original Image", use_column_width=True)

    cleaned = remove_blue_lines(image_np)

    if mode == "handwritten text":
        processed = preprocess_handwritten(cleaned)
    elif mode == "vehicle number plate":
        processed = preprocess_vehicle_plate(cleaned)
    else:
        processed = enhance_image(cleaned)

    st.image(processed, caption="🧪 Preprocessed Image", use_column_width=True, channels="GRAY")

    extracted_text = extract_text(processed, mode=mode)
    st.markdown("### ✅ Extracted Text:")
    st.text_area("🔍 OCR Output", extracted_text, height=300)

    if st.button("📥 Download as PDF"):
        pdf_bytes = generate_pdf(extracted_text)
        st.download_button("Download PDF", data=pdf_bytes, file_name="extracted_text.pdf", mime="application/pdf")
else:
    st.info("Upload an image to begin.")