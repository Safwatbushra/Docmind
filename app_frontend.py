import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocMind", page_icon="📄")
st.title("📄 DocMind — Bangla Document Q&A")
st.write("Upload a PDF and ask questions about it, in Bangla or English.")

# ---- Upload section ----
st.header("1. Upload a document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Upload and Index"):
        with st.spinner("Processing document... (this may take a few minutes for scanned PDFs)"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f"{API_URL}/upload", files=files)
            
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.success(f"Indexed {data['chunks_indexed']} chunks from {data['filename']}")
            else:
                st.error(f"Upload failed: {response.text}")

st.divider()

# ---- Question section ----
st.header("2. Ask a question")
question = st.text_input("Type your question here (Bangla or English)")
top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Thinking..."):
            payload = {"question": question, "top_k": top_k}
            response = requests.post(f"{API_URL}/ask", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                st.subheader("Answer")
                st.write(data["answer"])
                
                with st.expander("Sources used"):
                    for i, src in enumerate(data["sources"], 1):
                        st.markdown(f"**{i}.** {src}")
            else:
                st.error(f"Request failed: {response.text}")