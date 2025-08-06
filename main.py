from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import (
    extract_transactions,
    categorize_transactions,
    analyze_finances,
    generate_narrative,
    detect_red_flags
)
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file: UploadFile) -> str:
    """Extract text from PDF using pdfplumber and OCR fallback"""
    content = file.file.read()
    
    # Try text extraction first
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if len(text) > 100:  # Reasonable text length
            return text
    except:
        pass
    
    # Fallback to OCR for image-based PDFs
    images = convert_from_bytes(content)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image) + "\n"
    return text

def extract_text_from_image(file: UploadFile) -> str:
    """Extract text from image using OCR"""
    image = Image.open(io.BytesIO(file.file.read()))
    return pytesseract.image_to_string(image)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), months: int = 6):
    # Determine file type and extract text
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith(('.png', '.jpg', '.jpeg')):
        text = extract_text_from_image(file)
    else:
        raise HTTPException(400, "Unsupported file format")
    
    # GenAI Processing Pipeline
    transactions = extract_transactions(text)
    categorized = categorize_transactions(transactions)
    analysis = analyze_finances(categorized)
    narrative = generate_narrative(analysis)
    red_flags = detect_red_flags(transactions)
    
    return {
        "income": analysis.get("income", {}),
        "expenses": analysis.get("expenses", {}),
        "debt": analysis.get("debt", {}),
        "risk": analysis.get("risk", {}),
        "surplus": analysis.get("surplus", 0),
        "narrative": narrative,
        "red_flags": red_flags
    }

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Bank Statement Analyzer API is running"}