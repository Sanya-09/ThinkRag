import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url=None, collection="docs", dim=384, local_path=None):

        resolved_url = url or os.getenv("QDRANT_URL")
        resolved_local_path = local_path or os.getenv("QDRANT_LOCAL_PATH", "qdrant_storage")

        if resolved_url:
            self.client = QdrantClient(url=resolved_url, timeout=30)
        else:
            self.client = QdrantClient(path=resolved_local_path)

        self.collection = collection

        try:
            self.client.get_collection(self.collection)
        except:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector, top_k=5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        contexts = []
        sources = set()

        for r in results.points:
            payload = r.payload or {}

            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources)
        }