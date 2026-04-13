## Production Grade RAG Python App (No Docker)

This project is now configured to run fully local by default.

- Vector DB uses embedded Qdrant storage at `qdrant_storage/`.
- Streamlit ingests PDFs and answers questions directly (no Inngest event server required).

## 1. Create environment and install dependencies

A. For mac and linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install streamlit qdrant-client sentence-transformers llama-index llama-index-readers-file python-dotenv openai PyPDF2
```

B. For windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install streamlit qdrant-client sentence-transformers llama-index llama-index-readers-file python-dotenv openai PyPDF2
```
## 2. Configure environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini

# Optional: use remote Qdrant instead of local embedded mode
# QDRANT_URL=http://localhost:6333
# QDRANT_LOCAL_PATH=qdrant_storage
```

Notes:
- `OPENAI_API_KEY` is required for embeddings.
- If `OPENAI_API_KEY` is missing, answer generation falls back to context preview, but ingestion/query embeddings will still need the key.

## 3. Dependencies to install

A. For mac and linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install streamlit
pip install qdrant-client
pip install sentence-transformers
pip install llama-index
pip install llama-index-readers-file
pip install python-dotenv
pip install openai
pip install PyPDF2
```

B. For windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip

pip install streamlit
pip install qdrant-client
pip install sentence-transformers
pip install llama-index
pip install llama-index-readers-file
pip install python-dotenv
pip install openai
pip install PyPDF2
```


## 4. Run the Streamlit app

A. For mac and linux
```bash
rm -rf qdrant_storage
streamlit run streamlit_app.py
```

B. For windows
```bash
rmdir /s /q qdrant_storage
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit, upload a PDF, and ask questions.

## Optional: Run FastAPI app

```bash
uvicorn main:app --reload
```

The FastAPI app still contains the Inngest function definitions, but the Streamlit app no longer depends on Inngest for local usage.

WORKFLOW
PDF → Chunking → Embeddings → Qdrant → Retrieval → Answer
