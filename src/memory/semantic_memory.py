import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class SemanticMemory:
    """
    Vector-based semantic memory using Chroma + OpenAI embeddings.
    Stores interactions and retrieves by semantic similarity.
    """

    COLLECTION_PREFIX = "lab17_semantic"

    def __init__(self, session_id: str, persist_dir: str = None):
        self.session_id = session_id
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._collection_name = f"{self.COLLECTION_PREFIX}_{session_id}"
        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
        )

    def add_interaction(self, human: str, ai: str, metadata: dict = None) -> None:
        """Embed and store an interaction as a document."""
        text = f"Human: {human}\nAI: {ai}"
        doc = Document(
            page_content=text,
            metadata={
                "session_id": self.session_id,
                "human": human,
                "ai": ai,
                **(metadata or {}),
            },
        )
        self._store.add_documents([doc])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top_k semantically similar past interactions."""
        results = self._store.similarity_search_with_relevance_scores(query, k=top_k)
        output = []
        for doc, score in results:
            output.append({
                "human": doc.metadata.get("human", ""),
                "ai": doc.metadata.get("ai", ""),
                "score": round(score, 4),
                "content": doc.page_content,
            })
        return output

    def format_results(self, results: list[dict]) -> str:
        if not results:
            return ""
        lines = []
        for r in results:
            lines.append(f"Human: {r['human']}")
            lines.append(f"AI: {r['ai']}")
        return "\n".join(lines)

    def search_formatted(self, query: str, top_k: int = 3) -> str:
        return self.format_results(self.search(query, top_k))

    def count(self) -> int:
        return self._store._collection.count()

    def clear(self) -> None:
        self._store.delete_collection()
