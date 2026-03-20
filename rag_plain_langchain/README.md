# Plain RAG (LangChain + VectorDB) — Cricket Docs

This folder shows a **standard RAG** implementation using:

- **LangChain** for document loading, chunking, and the retrieval chain
- **Chroma** as the local VectorDB (persistent on disk)
- **Ollama** for embeddings + chat model

It uses the cricket knowledge base from:
- `AI_practice/rag_mcp/cricket_docs/`

## What you get

- `build_index.py`
  - Loads `cricket_docs/*.md`
  - Chunks documents
  - Creates embeddings via Ollama
  - Writes a persistent Chroma index to `./chroma_db/`

- `chatbot.py`
  - Gradio UI
  - Retrieves top-k chunks from Chroma
  - Calls the LLM with retrieved context
  - Shows sources (doc filenames)

## Prerequisites

1. Ollama running:

```bash
ollama serve
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

2. Install deps from `AI_practice/`:

```bash
pip install -r requirements.txt
```

## Build / rebuild the VectorDB

From this directory:

```bash
python build_index.py
```

## Run

```bash
python chatbot.py
```

## Try prompts

- "What is a no-ball?"
- "Give me some cricket terminologies"
- "From the example scorecard, who won and by how many runs?"

## Notes

- If you edit the markdown docs, rebuild the index.
- This is still a learning demo: it uses simple chunking and a basic prompt.


