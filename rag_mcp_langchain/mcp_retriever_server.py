import json
import os
import shutil
import re
from datetime import date

from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


HERE = os.path.dirname(__file__)
CRICKET_DOCS_DIR = os.path.abspath(os.path.join(HERE, "..", "rag_mcp", "cricket_docs"))
PERSIST_DIR = os.path.join(HERE, "chroma_db")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")


mcp = FastMCP(
    "langchain-chroma-retriever",
    instructions="MCP retriever server over a Chroma vector DB built from cricket_docs markdown.",
)


def _build_vectorstore(*, reset: bool) -> None:
    if not os.path.isdir(CRICKET_DOCS_DIR):
        raise RuntimeError(f"Cricket docs directory not found: {CRICKET_DOCS_DIR}")

    if reset and os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    loader = DirectoryLoader(
        CRICKET_DOCS_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="cricket_docs",
    )


def _load_vectorstore() -> Chroma:
    if not os.path.isdir(PERSIST_DIR):
        raise RuntimeError(
            f"Vector DB not found at {PERSIST_DIR}. Call the MCP tool index_cricket_docs(reset=true) or run this server once after indexing."
        )

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="cricket_docs",
    )


VECTORSTORE: Chroma | None = None
RETRIEVER = None


def _reload_retriever() -> None:
    global VECTORSTORE, RETRIEVER
    VECTORSTORE = _load_vectorstore()
    RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 5})


def _ensure_retriever_loaded() -> None:
    global RETRIEVER
    if RETRIEVER is None:
        _reload_retriever()


@mcp.tool()
def index_cricket_docs(reset: bool = False) -> str:
    """(Re)index the cricket_docs directory into a local Chroma DB.

    Args:
        reset: If true, deletes the existing Chroma DB directory and rebuilds.
    """

    _build_vectorstore(reset=bool(reset))
    _reload_retriever()

    return json.dumps(
        {
            "ok": True,
            "docs_dir": CRICKET_DOCS_DIR,
            "persist_dir": PERSIST_DIR,
            "collection": "cricket_docs",
            "reset": bool(reset),
        },
        ensure_ascii=False,
    )


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression.

    Supports numbers, whitespace, and operators: + - * / ( ) .
    """

    expr = (expression or "").strip()
    if not expr:
        return "Error: empty expression"

    if not re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.]+", expr):
        return "Error: only numbers and + - * / ( ) . are allowed"

    try:
        value = eval(expr, {"__builtins__": {}}, {})
    except Exception as e:
        return f"Error: {e}"

    return str(value)


@mcp.tool()
def calendar_list_events(days: int = 7) -> str:
    """List upcoming calendar events (demo tool with fixed sample data)."""

    d = max(1, int(days))
    today = date.today().isoformat()
    return json.dumps(
        {
            "today": today,
            "range_days": d,
            "events": [
                {"date": today, "title": "Demo: Cricket practice"},
                {"date": today, "title": "Demo: Team meeting"},
            ],
        },
        ensure_ascii=False,
    )


@mcp.tool()
def web_search(query: str) -> str:
    """Web search (demo stub). Does not make external network calls."""

    q = (query or "").strip()
    return json.dumps(
        {
            "query": q,
            "note": "web_search is a demo stub in this project and does not make external network calls.",
            "results": [],
        },
        ensure_ascii=False,
    )


@mcp.tool()
def search(query: str, top_k: int = 5) -> str:
    """Search the vector DB for relevant chunks.

    Args:
        query: Natural language query.
        top_k: Number of chunks to return.
    """

    try:
        _ensure_retriever_loaded()
    except Exception as e:
        raise RuntimeError(
            "Retriever is not ready. Run the MCP tool index_cricket_docs(reset=true) to build the VectorDB first. "
            f"Details: {e}"
        ) from e

    # For newer LangChain retrievers (Runnable interface)
    if hasattr(RETRIEVER, "invoke"):
        docs = RETRIEVER.invoke(query)
    else:
        docs = RETRIEVER.get_relevant_documents(query)

    hits = []
    for d in docs[: max(1, int(top_k))]:
        src = d.metadata.get("source", "")
        doc_id = os.path.basename(src) if src else ""
        text = d.page_content
        snippet = text[:500] + ("..." if len(text) > 500 else "")
        hits.append({"doc_id": doc_id, "source": src, "snippet": snippet})

    return json.dumps({"query": query, "top_k": top_k, "hits": hits}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
