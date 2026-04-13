from pathlib import Path
import uuid

import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


@st.cache_resource
def get_db():
    return QdrantStorage()


db = get_db()


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


def ingest_pdf_local(pdf_path: Path, db) -> int:
    source_id = pdf_path.name
    chunks = load_and_chunk_pdf(str(pdf_path.resolve()))

    if not chunks:
        return 0

    vectors = embed_texts(chunks)

    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

    db.upsert(ids, vectors, payloads)

    return len(chunks)


def answer_question_local(question: str, top_k: int, db):
    query_vec = embed_texts([question])[0]

    found = db.search(query_vec, top_k)
    contexts = found["contexts"]
    sources = found["sources"]

    if not contexts:
        return "No relevant context found in the ingested PDFs.", sources

    client = get_openai_client()

    if client is None:
        preview = "\n\n".join(f"- {c}" for c in contexts[:3])
        return (
            "OPENAI_API_KEY is not set. Showing top matching context instead:\n\n"
            f"{preview}",
            sources,
        )

    context_block = "\n\n".join(f"- {c}" for c in contexts)

    try:
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {
                    "role": "user",
                    "content": (
                        "Use the following context to answer the question.\n\n"
                        f"Context:\n{context_block}\n\n"
                        f"Question: {question}\n"
                        "Answer concisely using the context above."
                    ),
                },
            ],
        )

        answer = completion.choices[0].message.content or ""

    except Exception:
        # ✅ fallback when API fails
        answer = "⚠️ Showing best matching content from your PDF:\n\n"
        answer += "\n\n".join(f"- {c}" for c in contexts[:3])

    return answer.strip(), sources


st.title("📄 Upload a PDF to Ingest")

uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded is not None:
    with st.spinner("Uploading and ingesting..."):
        path = save_uploaded_pdf(uploaded)
        ingested_count = ingest_pdf_local(path, db)

    st.success(f"Ingested {ingested_count} chunks from: {path.name}")

st.divider()

st.title("💬 Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("Chunks to retrieve", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Generating answer..."):
            answer, sources = answer_question_local(question.strip(), int(top_k), db)

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")