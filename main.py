from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import requests
from pypdf import PdfReader

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

app = FastAPI(title="DocMind RAG API")

# ---- Startup: load model and build index once ----
print("Loading embedding model...")
model = SentenceTransformer("intfloat/multilingual-e5-base")

def extract_pdf_text(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]

print("Extracting and indexing test.pdf...")
text = extract_pdf_text("test.pdf")
chunks = chunk_text(text)

client = chromadb.Client()
collection = client.create_collection("pdf_docs")
embeddings = model.encode(chunks, normalize_embeddings=True)
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(chunks))]
)
print(f"Indexed {len(chunks)} chunks. Ready.")

# ---- Request/response schemas ----
class Question(BaseModel):
    question: str
    top_k: int = 3

class Answer(BaseModel):
    answer: str
    sources: list[str]

# ---- Endpoint ----
@app.post("/ask", response_model=Answer)
def ask(payload: Question):
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
import re

def extract_pdf_text(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def clean_text(text):
    # Remove repeated watermark/spam lines
    lines = text.split("\n")
    cleaned_lines = [
        line for line in lines
        if "teachingbd" not in line.lower()
    ]
    return "\n".join(cleaned_lines)