from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)

    texts = [d.text for d in docs if getattr(d, "text", None)]

    chunks = []
    for text in texts:
        split_chunks = splitter.split_text(text)
        chunks.extend(split_chunks)

    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=32,        
        show_progress_bar=False
    )

    return embeddings.tolist()

def debug_embedding_dimension():
    test = model.encode(["test"])
    print("Embedding dimension:", len(test[0]))