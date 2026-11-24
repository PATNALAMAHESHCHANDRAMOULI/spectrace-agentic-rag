import os
from pathlib import Path
import chromadb
from pypdf import PdfReader
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import get_settings


def get_embedding_model() -> HuggingFaceEmbedding:
    """Load the local Hugging Face embedding model defined in config."""
    settings = get_settings()
    # Set HF_TOKEN in environment if provided so Hugging Face Hub downloads are authenticated
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
    return HuggingFaceEmbedding(model_name=settings.embedding_model_name)


def load_pdf_pages(pdf_dir: Path) -> list[Document]:
    """Extract clean text page-by-page using pypdf with exact citation metadata."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{pdf_dir}'. Place your datasheets there first."
        )

    documents: list[Document] = []
    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            # Skip blank or image-only pages that have no extractable text
            if not text:
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": pdf_path.name,
                        "page_label": str(page_idx),
                    },
                )
            )
    return documents


def ingest_documents(reset_existing: bool = True) -> int:
    """Parse PDFs, chunk with metadata, persist vectors to ChromaDB, and save nodes for BM25."""
    settings = get_settings()
    pdf_dir = Path(settings.raw_pdf_dir)

    print(f"[1/4] Extracting text page-by-page from '{pdf_dir}'...")
    documents = load_pdf_pages(pdf_dir)
    print(f"      Loaded {len(documents)} readable PDF pages.")

    print(
        f"[2/4] Chunking pages (chunk_size={settings.chunk_size}, "
        f"chunk_overlap={settings.chunk_overlap})..."
    )
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"      Created {len(nodes)} clean text chunks.")

    print(f"[3/4] Initializing ChromaDB at '{settings.chroma_persist_dir}'...")
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    if reset_existing:
        existing_names = [c.name for c in chroma_client.list_collections()]
        if settings.chroma_collection_name in existing_names:
            chroma_client.delete_collection(name=settings.chroma_collection_name)

    chroma_collection = chroma_client.get_or_create_collection(
        name=settings.chroma_collection_name
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Save nodes to a local JSON docstore so BM25 can load them instantly without re-parsing PDFs
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(persist_path=settings.docstore_path)

    print(f"[4/4] Embedding {len(nodes)} chunks with '{settings.embedding_model_name}'...")
    embed_model = get_embedding_model()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(
        f"\nIngestion complete! Stored {chroma_collection.count()} vectors in ChromaDB "
        f"and saved BM25 docstore to '{settings.docstore_path}'."
    )
    return chroma_collection.count()


if __name__ == "__main__":
    ingest_documents(reset_existing=True)