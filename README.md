

```markdown
# DocMind

A RAG-based document Q&A system for Bangla text. Most LLM tooling is built around English, so I wanted to see what it'd take to build something that actually works well for Bangla documents.

Right now the core pipeline and FastAPI backend are working. Docker/Kubernetes setup is next.

## How it works

PDF → text extraction (PyPDF) → chunking → embeddings (multilingual-e5-base) → stored in ChromaDB.
When you ask a question, it embeds the query, retrieves the most relevant chunks, and passes them to Gemini to generate a grounded answer instead of letting the model guess.

## Stack

Python, FastAPI, ChromaDB, multilingual-e5-base, Google Gemini API, PyPDF

## Running it locally

```bash
git clone https://github.com/Safwatbushra/docmind.git
cd docmind
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

Add a `.env` file with your Gemini key:
```
GEMINI_API_KEY=your_key_here
```

Then run:
```bash
uvicorn main:app --reload
```

Swagger docs will be at `http://127.0.0.1:8000/docs`.

## Example

```bash
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@document.pdf"

curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "your question here"}'

D
