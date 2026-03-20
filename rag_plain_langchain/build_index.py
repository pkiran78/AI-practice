import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


HERE = os.path.dirname(__file__)
CRICKET_DOCS_DIR = os.path.abspath(os.path.join(HERE, "..", "rag_mcp", "cricket_docs"))
PERSIST_DIR = os.path.join(HERE, "chroma_db")

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def build_index(*, reset: bool = False) -> None:
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

    print(f"Indexed {len(docs)} docs into {len(chunks)} chunks")
    print(f"Chroma persisted to: {PERSIST_DIR}")


if __name__ == "__main__":
    build_index(reset=True)
