from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import requests
from pypdf import PdfReader

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

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

pdf_path = "test.pdf"
text = extract_pdf_text(pdf_path)
chunks = chunk_text(text)
print(f"Extracted {len(text)} characters, split into {len(chunks)} chunks")

client = chromadb.Client()
collection = client.create_collection("pdf_docs")
embeddings = model.encode(chunks, normalize_embeddings=True)
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(chunks))]
)

def ask(query: str, top_k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=top_k)
    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={API_KEY}",
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    data = response.json()

    try:
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        answer = f"Error or unexpected response: {data}"

    return answer, retrieved_chunks


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question about the PDF (or 'quit'): ")
        if question.lower() == "quit":
            break
        answer, sources = ask(question)
        print("\nAnswer:", answer)