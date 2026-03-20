# RAG via MCP (Learning Demo)

This folder demonstrates a small Retrieval-Augmented Generation (RAG) pipeline where:

- The **LLM + UI** live in the client (Gradio + Ollama).
- The **Retriever / Index** lives in a separate **MCP server** (FastMCP).

It is intentionally lightweight (pure Python text retrieval) so you can focus on the architecture and the tool-calling loop.

## What you get

- `mcp_rag_server.py`
  - MCP tools:
    - `index_directory(directory: str, glob_pattern: str = "*.md")`
    - `search(query: str, top_k: int = 5)`
  - Stores an in-memory TF-IDF-ish index (token -> document frequencies).

- `rag_mcp_client_chatbot.py`
  - Gradio chat UI
  - Uses Ollama via OpenAI-compatible API
  - Discovers MCP tools and calls `search()` when it needs context
  - Builds an answer with **citations** (doc ids)

- `sample_docs/`
  - A few small `.txt` files to retrieve from.

- `cricket_docs/`
  - A small cricket knowledge base in `.md` format for learning RAG.

## Prerequisites

1. Start Ollama and pull a model:

```bash
ollama serve
ollama pull qwen2.5:3b
```

2. Install Python deps (from repo root):

```bash
pip install -r ..\requirements.txt
```

## Run

From this directory:

```bash
python rag_mcp_client_chatbot.py
```

By default the client indexes:

- `cricket_docs\*.md` (if the folder exists)
- otherwise falls back to `sample_docs\*.txt`

Notes:
- The MCP server is started automatically by `fastmcp.Client` (same pattern you used in `mcp_approach/`).
- If you want to run the server manually:

```bash
python mcp_rag_server.py
```

## Try prompts

- "What is RAG? Answer with sources."
- "Explain chunking and why overlap helps."
- "How does MCP help in a RAG system?"
- "From the example scorecard, who won and by how many runs? Answer with sources."
- "What is the difference between a wide and a no-ball? Provide sources."

## Next experiments

- Add more `.md` docs into `cricket_docs/` and ask questions.
- Improve chunking (split by headings/paragraphs).
- Add reranking (LLM or heuristic).
- Persist the index to disk.
