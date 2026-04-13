import os
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()

# Initialize vector DB
db = QdrantStorage()


# 🔹 STEP 1: Ingest PDF (store embeddings)
def ingest_pdf(pdf_path: str):
    chunks = load_and_chunk_pdf(pdf_path)

    vectors = embed_texts(chunks)

    ids = [str(i) for i in range(len(chunks))]
    payloads = [{"text": chunks[i], "source": pdf_path} for i in range(len(chunks))]

    db.upsert(ids, vectors, payloads)

    return {"status": "success", "chunks": len(chunks)}


# 🔹 STEP 2: Query function (MAIN RAG LOGIC)
def answer_query(question: str, top_k: int = 5):
    # Convert question → embedding
    query_vec = embed_texts([question])[0]

    # Retrieve relevant chunks
    results = db.search(query_vec, top_k)

    contexts = results["contexts"]
    sources = results["sources"]

    # Combine context
    context_text = "\n\n".join(contexts)

    # Create prompt
    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """

    # Call OpenAI
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer only from given context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "sources": sources
    }
