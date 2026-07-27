from sentence_transformers import SentenceTransformer
import chromadb

# Load a multilingual embedding model (works for Bangla + English)
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Read and chunk the document (simple: split by paragraph)
with open("sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
print(f"Loaded {len(chunks)} chunks")

# Set up Chroma (local vector database)
client = chromadb.Client()
collection = client.create_collection("docs")

# Embed and store each chunk
embeddings = model.encode(chunks, normalize_embeddings=True)
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(chunks))]
)

# Test query
query = "When did Bangladesh become independent?"
query_embedding = model.encode([query], normalize_embeddings=True)
results = collection.query(query_embeddings=query_embedding.tolist(), n_results=2)

print("\nTop matches:")
for doc in results["documents"][0]:
    print("-", doc[:100])