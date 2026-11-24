from dataclasses import dataclass
from pathlib import Path
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import get_settings
from src.rag.ingestion import get_embedding_model


@dataclass(frozen=True)
class RetrievedChunk:
    """Clean, typed retrieval result with source citation metadata."""

    text: str
    file_name: str
    page_label: str
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.file_name}, p. {self.page_label}]"


class HybridRetriever:
    """Two-stage Hybrid RAG Retriever: (Vector + BM25 via RRF) -> Cross-Encoder Reranker."""

    def __init__(self, candidate_k: int = 10, final_top_n: int = 4) -> None:
        settings = get_settings()
        self.candidate_k = candidate_k
        self.final_top_n = final_top_n

        if not Path(settings.docstore_path).exists():
            raise FileNotFoundError(
                f"Docstore not found at '{settings.docstore_path}'. "
                "Run `python -m src.rag.ingestion` first."
            )

        # 1. Dense Vector Retriever (ChromaDB + local Hugging Face embeddings)
        chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        chroma_collection = chroma_client.get_collection(
            name=settings.chroma_collection_name
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        embed_model = get_embedding_model()

        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        self.vector_retriever = vector_index.as_retriever(
            similarity_top_k=self.candidate_k
        )

        # 2. Sparse Keyword Retriever (BM25 over persisted nodes)
        docstore = SimpleDocumentStore.from_persist_path(settings.docstore_path)
        all_nodes = list(docstore.docs.values())
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=self.candidate_k,
        )

        # 3. Stage-2 Cross-Encoder Reranker
        self.reranker = SentenceTransformerRerank(
            model=settings.reranker_model_name,
            top_n=self.final_top_n,
        )

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[NodeWithScore]],
        k: int = 60,
    ) -> list[NodeWithScore]:
        """Fuse multiple ranked node lists using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}

        for results in ranked_lists:
            for rank, item in enumerate(results, start=1):
                node_id = item.node.node_id
                rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (k + rank))
                if node_id not in node_map:
                    node_map[node_id] = item

        sorted_ids = sorted(rrf_scores, key=lambda nid: rrf_scores[nid], reverse=True)
        return [
            NodeWithScore(node=node_map[nid].node, score=rrf_scores[nid])
            for nid in sorted_ids
        ]

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Run Vector + BM25 -> RRF fusion -> Cross-Encoder reranking."""
        vector_nodes = self.vector_retriever.retrieve(query)
        bm25_nodes = self.bm25_retriever.retrieve(query)

        fused_nodes = self._reciprocal_rank_fusion([vector_nodes, bm25_nodes])
        reranked_nodes = self.reranker.postprocess_nodes(fused_nodes, query_str=query)

        return [
            RetrievedChunk(
                text=item.node.get_content().strip(),
                file_name=str(item.node.metadata.get("file_name", "unknown.pdf")),
                page_label=str(item.node.metadata.get("page_label", "?")),
                score=float(item.score or 0.0),
            )
            for item in reranked_nodes
        ]

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks with explicit source citations for LLM grounding."""
        if not chunks:
            return "No relevant documentation found."
        blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            blocks.append(
                f"--- Evidence {idx} {chunk.citation} (relevance={chunk.score:.3f}) ---\n"
                f"{chunk.text}"
            )
        return "\n\n".join(blocks)


if __name__ == "__main__":
    retriever = HybridRetriever(candidate_k=10, final_top_n=3)

    test_queries = [
        "ESP32-S3-WROOM-1 deep sleep current consumption uA",
        "STM32F401 Standby mode current consumption at 3.3V",
    ]

    for q in test_queries:
        print(f"\n==================== QUERY: {q} ====================")
        results = retriever.retrieve(q)
        for i, chunk in enumerate(results, start=1):
            preview = chunk.text.replace("\n", " ")[:220]
            print(f"[{i}] {chunk.citation} | score={chunk.score:.4f}")
            print(f"    {preview}...")