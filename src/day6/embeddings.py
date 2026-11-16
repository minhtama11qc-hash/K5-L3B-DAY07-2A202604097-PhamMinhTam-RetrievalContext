"""Real embedding provider: Sentence Transformers running locally (no API key, cost 0)."""
from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class SentenceTransformerEmbedder:
    """Local Sentence Transformers model.

    Vectors are L2-normalized so cosine distance in pgvector is well defined. The same
    encoder is used for documents and queries (all-MiniLM-L6-v2 is a symmetric model and
    needs no query/document prefix). Errors are raised, never replaced by mock vectors.
    """

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dimensions = int(get_dim())
        self.max_seq_length = int(self.model.max_seq_length)

    def token_count(self, text: str) -> int:
        """Word-piece tokens (without special tokens), used for size/truncation stats."""
        return len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
        if len(vectors) != len(texts) or vectors.shape[1] != self.dimensions:
            raise RuntimeError("embedding output shape does not match input")
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts) if texts else []

    def embed_query(self, question: str) -> list[float]:
        return self._encode([question])[0]


# Name kept from the starter skeleton.
MyEmbedder = SentenceTransformerEmbedder
