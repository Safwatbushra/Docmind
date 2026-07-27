from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import requests
from pypdf import PdfReader
import io
import re
import pytesseract
from pdf2image import convert_from_bytes

# ---- OCR tool paths (Windows-specific) ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\user\Downloads\Release-24.08.0-0\bin"

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

app = FastAPI(title="DocMind RAG API")

print("Loading embedding model...")
model = SentenceTransformer("intfloat/multilingual-e5-base")

client = chromadb.Client()
collection = None


def extract_pdf_text(file_bytes):
    """Try normal text extraction first (fast, works for text-based PDFs)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


def extract_text_with_ocr(file_bytes):
    """Fallback: convert PDF pages to images, then OCR them (Bangla + English)."""
    images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
    full_text = ""
    for i, img in enumerate(images):
        print(f"OCR processing page {i+1}/{len(images)}...")
        page_text = pytesseract.image_to_string(img, lang="ben+eng")
        full_text += page_text + "\n"
    return full_text


def clean_text(text):
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r'(https?://\S+|www\.\S+)', stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(stripped)

    deduped = []
    for line in cleaned_lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    return "\n".join(deduped)


def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global collection

    file_bytes = await file.read()

    text = extract_pdf_text(file_bytes)
    print(f"DEBUG: Normal extraction length: {len(text.strip())}")

    # If normal extraction found little/no text, fall back to OCR
    if len(text.strip()) < 50:
        print("DEBUG: Falling back to OCR...")
        text = extract_text_with_ocr(file_bytes)
        print(f"DEBUG: OCR extraction length: {len(text.strip())}")

    text = clean_text(text)
    chunks = chunk_text(text)
    print(f"DEBUG: Number of chunks: {len(chunks)}")

    if not chunks:
        return {"error": "No text could be extracted from this PDF, even with OCR."}

    try:
        client.delete_collection("pdf_docs")
    except Exception:
        pass
    collection = client.create_collection("pdf_docs")

    embeddings = model.encode(chunks, normalize_embeddings=True)
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(chunks))]
    )

    return {"filename": file.filename, "chunks_indexed": len(chunks)}


class Question(BaseModel):
    question: str
    top_k: int = 3


class Answer(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=Answer)
def ask(payload: Question):
    if collection is None:
        return Answer(answer="No document uploaded yet. Please upload a PDF first.", sources=[])

    query_embedding = model.encode([payload.question], normalize_embeddings=True)
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=payload.top_k)
    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {payload.question}

Answer:"""

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={API_KEY}",
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    data = response.json()

    try:
        answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        answer_text = f"Error: {data}"

    return Answer(answer=answer_text, sources=retrieved_chunks)


@app.get("/")
def health():
    return {"status": "ok"}