# RAG (LangChain + Chroma) via MCP — Learning Demo

This folder is the **MCP version** of `rag_plain_langchain/`.

Conceptually, nothing about RAG changes. The only change is **where retrieval runs**:

- **Client**: Gradio UI + Ollama LLM (answers)
- **Server**: MCP tool server that owns the VectorDB (Chroma) and performs retrieval

## Files

- `mcp_retriever_server.py`
  - Loads the existing Chroma DB from `../rag_plain_langchain/chroma_db`
  - Exposes MCP tool:
    - `search(query: str, top_k: int = 5) -> str`

- `mcp_client_chatbot.py`
  - Gradio chat UI
  - Calls MCP `search()` tool
  - Injects retrieved context into the LLM prompt
  - Appends a `Sources:` list

## Prerequisites

1. Ensure Ollama is running and models are available:

2. Start Ollama + pull models:

```bash
ollama serve
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

3. Install deps (from `AI_practice/`):

```bash
pip install -r requirements.txt
```

## Run

From this directory:

```bash
python mcp_client_chatbot.py
```

## Agentic demo (LLM chooses tools)

This project also includes an agentic client where the LLM can choose among multiple tools:

- MCP `search` (retrieval)
- `calculator`
- `calendar_list_events` (demo data)
- `web_search` (stub, no external network calls)

Run:

```bash
python mcp_agentic_chatbot.py
```

## Build / rebuild the VectorDB (from this folder)

This MCP version stores its own Chroma DB at:
- `rag_mcp_langchain/chroma_db/`

Option A (recommended): run the server once and call the MCP tool `index_cricket_docs`.

Option B: run the server manually and then use any MCP client to call:

- `index_cricket_docs(reset=true)`

Notes:
- The MCP server is started automatically by `fastmcp.Client`.
- If you want to run server manually:

```bash
python mcp_retriever_server.py
```

## Try prompts

- "What is a no-ball?"
- "Give me some cricket terminologies"
- "From the example scorecard, who won and by how many runs?"

## Learning takeaway

- Plain RAG: `retriever.search(...)` is a local function call.
- MCP RAG: `search(...)` is a tool call over MCP.

The prompt and grounding logic are identical.
