"""
src/retrieve.py — Knowledge Retrieval & Vector Search Layer with LangGraph Integration

Satisfies Acceptance Criterion A4:
Retrieval runs against the supplied documentation corpus and returns identifiable
source passages that resolve back to real passages in the corpus, not invented references.
Applies a relevance threshold and returns nothing rather than something irrelevant.
"""

import os
import json
import logging
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress external library noise (HuggingFace Hub, HTTP requests, progress bars)
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.simplefilter("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

for noisy_logger in [
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "openai",
    "urllib3",
    "chromadb",
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
]:
    nl = logging.getLogger(noisy_logger)
    nl.setLevel(logging.ERROR)
    nl.propagate = False
    nl.handlers.clear()

try:
    import huggingface_hub.utils
    huggingface_hub.utils.disable_progress_bars()
except Exception:
    pass

try:
    import transformers.utils.logging
    transformers.utils.logging.disable_progress_bar()
    transformers.utils.logging.set_verbosity_error()
except Exception:
    pass

# Default configurations
DEFAULT_CHROMA_PATH = os.getenv("CHROMA_PATH", "./storage/chroma")
DEFAULT_DOCS_PATH = os.getenv("DOCS_PATH", "./data/documentation.json")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.40"))
COLLECTION_NAME = "cloudserve_kb"


class Passage(TypedDict):
    doc_id: str
    title: str
    category: str
    content: str
    score: float
    chunk_id: str


class SupportState(TypedDict, total=False):
    """Workflow state for retrieval node and retrieval StateGraph."""
    ticket_id: str
    clean_text: str
    body: str
    subject: str
    chroma_path: str
    retrieval_threshold: float
    retrieval_top_k: int
    retrieved_passages: List[Dict[str, Any]]
    top_score: float
    has_relevant_docs: bool
    retrieval_error: Optional[str]





_client: Optional[chromadb.PersistentClient] = None
_embedding_fn = None


def get_embedding_function():
    global _embedding_fn
    if _embedding_fn is None:
        logger.info(f"Loading SentenceTransformer embedding function: {DEFAULT_EMBEDDING_MODEL}")
        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=DEFAULT_EMBEDDING_MODEL
        )
    return _embedding_fn


def get_chroma_client(chroma_path: str = DEFAULT_CHROMA_PATH) -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(chroma_path, exist_ok=True)
        _client = chromadb.PersistentClient(path=chroma_path)
    return _client


def build_knowledge_base_index(
    docs_path: str = DEFAULT_DOCS_PATH,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    force_rebuild: bool = False
) -> int:
    """
    Chunks and embeds the 29 documentation articles into Chroma vector store.
    Idempotent: Re-indexes only if empty or force_rebuild=True.
    """
    client = get_chroma_client(chroma_path)
    ef = get_embedding_function()

    # Locate documentation file
    path_obj = Path(docs_path)
    if not path_obj.exists():
        fallback = Path("docs/05_Datasets/documentation.json")
        if fallback.exists():
            path_obj = fallback
        else:
            raise FileNotFoundError(f"Documentation file not found at {docs_path} or {fallback}")

    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection '{COLLECTION_NAME}' for rebuild.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        logger.info(f"Chroma index already contains {existing_count} chunks. Skipping re-indexing.")
        return existing_count

    logger.info(f"Indexing documents from {path_obj} into Chroma...")
    with open(path_obj, "r", encoding="utf-8") as f:
        documents = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n## ", "\n\n", "\n", " "]
    )

    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for doc in documents:
        doc_id = doc["doc_id"]
        title = doc.get("title", "")
        category = doc.get("category", "")
        content = doc.get("content", "")

        chunks = splitter.split_text(content)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            ids.append(chunk_id)
            # Prefix passage with document title for optimal semantic grounding
            texts.append(f"Title: {title}\n\n{chunk}")
            metadatas.append({
                "doc_id": doc_id,
                "title": title,
                "category": category,
                "chunk_index": idx
            })

    # Batch insert into Chroma
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size]
        )

    logger.info(f"Successfully indexed {len(texts)} passages from {len(documents)} articles.")
    return len(texts)


def retrieve_passages(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    chroma_path: str = DEFAULT_CHROMA_PATH
) -> List[Passage]:
    """
    Performs vector similarity search for a customer query against Chroma index.
    Filters out passages below the similarity threshold.
    """
    if not query or not query.strip():
        return []

    client = get_chroma_client(chroma_path)
    ef = get_embedding_function()

    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception:
        # Auto-build index if missing
        logger.warning("Chroma collection missing. Building index now...")
        build_knowledge_base_index(chroma_path=chroma_path)
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    results = collection.query(
        query_texts=[query.strip()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    passages: List[Passage] = []
    if not results or not results["documents"] or not results["documents"][0]:
        return passages

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc_text, meta, dist in zip(docs, metas, distances):
        # In Chroma with cosine space: distance = 1 - cosine_similarity
        # Hence similarity_score = 1.0 - distance
        similarity = max(0.0, min(1.0, 1.0 - float(dist)))
        
        if similarity >= threshold:
            passages.append({
                "doc_id": meta.get("doc_id", "UNKNOWN"),
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "content": doc_text,
                "score": round(similarity, 4),
                "chunk_id": f"{meta.get('doc_id')}_chunk_{meta.get('chunk_index', 0)}"
            })

    return passages


# ==============================================================================
# LangGraph Node & Workflow Graph
# ==============================================================================

def retrieve_node(state: Dict[str, Any], chroma_path: Optional[str] = None) -> Dict[str, Any]:
    """
    LangGraph Node: Executes semantic retrieval for incoming ticket state.
    """
    query = state.get("clean_text") or state.get("body", "")
    if not query and state.get("subject"):
        query = state["subject"]

    active_chroma_path = chroma_path or state.get("chroma_path") or DEFAULT_CHROMA_PATH
    threshold = float(state.get("retrieval_threshold", DEFAULT_SIMILARITY_THRESHOLD))
    top_k = int(state.get("retrieval_top_k", DEFAULT_TOP_K))

    try:
        passages = retrieve_passages(
            query=query,
            top_k=top_k,
            threshold=threshold,
            chroma_path=active_chroma_path
        )
        top_score = passages[0]["score"] if passages else 0.0
        return {
            "retrieved_passages": passages,
            "top_score": top_score,
            "has_relevant_docs": len(passages) > 0,
            "retrieval_error": None
        }
    except Exception as e:
        logger.error(f"Error in retrieve_node: {e}")
        return {
            "retrieved_passages": [],
            "top_score": 0.0,
            "has_relevant_docs": False,
            "retrieval_error": str(e)
        }


def create_retrieval_graph():
    """
    Compiles a dedicated LangGraph StateGraph containing the retrieval pipeline.
    Can be run independently or embedded into the overarching support graph.
    """
    workflow = StateGraph(SupportState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", END)
    return workflow.compile()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CloudServe Knowledge Retrieval Layer")
    parser.add_argument("--index", action="store_true", help="Build or rebuild Chroma index")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild Chroma index")
    parser.add_argument("--query", type=str, help="Query string to retrieve")
    args = parser.parse_args()

    if args.index or args.rebuild:
        count = build_knowledge_base_index(force_rebuild=args.rebuild)
        print(f"Chroma index populated with {count} chunks.")

    if args.query:
        print(f"\nQuerying: '{args.query}'...")
        results = retrieve_passages(args.query)
        print(f"Retrieved {len(results)} relevant passages:")
        for idx, p in enumerate(results, 1):
            print(f"\n[{idx}] doc_id: {p['doc_id']} | Title: {p['title']} | Score: {p['score']}")
            print(f"Content snippet: {p['content'][:150]}...")
