import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP


mcp = FastMCP(
    "rag-retriever-server",
    instructions="Learning MCP server that provides simple RAG retrieval tools.",
)


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


@dataclass
class Doc:
    doc_id: str
    source_path: str
    text: str


class SimpleTfidfIndex:
    def __init__(self) -> None:
        self._docs: list[Doc] = []
        self._df: dict[str, int] = {}
        self._doc_tf: list[dict[str, int]] = []

    def clear(self) -> None:
        self._docs = []
        self._df = {}
        self._doc_tf = []

    @property
    def size(self) -> int:
        return len(self._docs)

    def add_document(self, *, doc_id: str, source_path: str, text: str) -> None:
        tokens = _tokenize(text)
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        for tok in tf.keys():
            self._df[tok] = self._df.get(tok, 0) + 1

        self._docs.append(Doc(doc_id=doc_id, source_path=source_path, text=text))
        self._doc_tf.append(tf)

    def _idf(self, term: str) -> float:
        # Smooth IDF
        n = max(len(self._docs), 1)
        df = self._df.get(term, 0)
        return math.log((n + 1) / (df + 1)) + 1.0

    def search(self, *, query: str, top_k: int) -> list[dict[str, Any]]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self._docs:
            return []

        q_tf: dict[str, int] = {}
        for t in q_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1

        # Build query weights
        q_w: dict[str, float] = {}
        for t, c in q_tf.items():
            q_w[t] = float(c) * self._idf(t)

        q_norm = math.sqrt(sum(v * v for v in q_w.values())) or 1.0

        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._doc_tf):
            # cosine similarity of tf-idf vectors, computed sparsely
            dot = 0.0
            d_norm_sq = 0.0

            for t, c in tf.items():
                w = float(c) * self._idf(t)
                d_norm_sq += w * w

            for t, qw in q_w.items():
                if t in tf:
                    dot += qw * (float(tf[t]) * self._idf(t))

            d_norm = math.sqrt(d_norm_sq) or 1.0
            score = dot / (q_norm * d_norm)
            if score > 0.0:
                scored.append((score, i))

        scored.sort(reverse=True, key=lambda x: x[0])
        hits = scored[: max(1, int(top_k))]

        results: list[dict[str, Any]] = []
        for score, idx in hits:
            d = self._docs[idx]
            snippet = d.text
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."

            results.append(
                {
                    "doc_id": d.doc_id,
                    "source_path": d.source_path,
                    "score": score,
                    "snippet": snippet,
                }
            )

        return results


INDEX = SimpleTfidfIndex()


@mcp.tool()
def index_directory(directory: str, glob_pattern: str = "*.txt") -> str:
    """Index all matching text files under a directory.

    Args:
        directory: Directory to read documents from.
        glob_pattern: Filename pattern like '*.txt'. Only basic suffix matching is used in this demo.
    """

    if not os.path.isdir(directory):
        return json.dumps({"error": f"Not a directory: {directory}"})

    # Basic pattern support: only '*.ext'
    ext = None
    if glob_pattern.startswith("*."):
        ext = glob_pattern[1:]  # '.txt'

    INDEX.clear()

    added = 0
    for root, _, files in os.walk(directory):
        for name in files:
            if ext and not name.endswith(ext):
                continue

            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                # Skip unreadable files
                continue

            doc_id = os.path.relpath(path, directory).replace("\\", "/")
            INDEX.add_document(doc_id=doc_id, source_path=path, text=text)
            added += 1

    return json.dumps({"indexed": added, "directory": directory})


@mcp.tool()
def search(query: str, top_k: int = 5) -> str:
    """Search the index for relevant documents.

    Args:
        query: Natural language query.
        top_k: Number of hits to return.
    """

    if INDEX.size == 0:
        return json.dumps({"error": "Index is empty. Call index_directory() first."})

    hits = INDEX.search(query=query, top_k=top_k)
    return json.dumps({"query": query, "top_k": top_k, "hits": hits})


if __name__ == "__main__":
    print("Starting MCP RAG Retriever Server: rag-retriever-server")
    print("Tools available: index_directory, search")
    mcp.run()
