import os

import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI


HERE = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(HERE, "chroma_db")

OLLAMA_OPENAI_BASE_URL = os.environ.get("OLLAMA_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "nomic-embed-text")


def _load_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    if not os.path.isdir(PERSIST_DIR):
        raise RuntimeError(
            f"Vector DB not found at {PERSIST_DIR}. Run: python build_index.py"
        )

    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="cricket_docs",
    )


VECTORSTORE = _load_vectorstore()
RETRIEVER = VECTORSTORE.as_retriever(search_kwargs={"k": 5})

LLM = ChatOpenAI(
    base_url=OLLAMA_OPENAI_BASE_URL,
    api_key="ollama",
    model=OLLAMA_MODEL,
    temperature=0.2,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about cricket using the provided context. "
    "Use only the provided context for factual claims, and cite sources as [source: <doc_id>]. "
    "If the context is insufficient, say you don't have enough information in the indexed docs."
)


def _format_sources(docs) -> str:
    sources = []
    for d in docs:
        # DirectoryLoader/TextLoader stores the file path in metadata['source']
        src = d.metadata.get("source") if hasattr(d, "metadata") else None
        if not src:
            continue
        sources.append(os.path.basename(src))

    # dedupe while preserving order
    seen = set()
    ordered = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return "\n".join(f"- {s}" for s in ordered) if ordered else "- (no sources)"


def answer(message: str, history: list):
    # Gradio 6.x Chatbot expects messages format: list of {role, content}
    if hasattr(RETRIEVER, "invoke"):
        retrieved_docs = RETRIEVER.invoke(message)
    else:
        retrieved_docs = RETRIEVER.get_relevant_documents(message)

    context_blocks = []
    for d in retrieved_docs:
        src = os.path.basename(d.metadata.get("source", ""))
        context_blocks.append(f"[source: {src}]\n{d.page_content}")

    context = "\n\n".join(context_blocks)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"User question: {message}"
    )

    resp = LLM.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    text += "\n\nSources:\n" + _format_sources(retrieved_docs)

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": text})
    return "", history


with gr.Blocks(title="Plain RAG (LangChain + Chroma)") as demo:
    gr.Markdown(
        "# Plain RAG (LangChain + VectorDB)\n"
        "This version uses LangChain + Chroma. Run `python build_index.py` first."
    )

    chatbot = gr.Chatbot(height=520)
    msg = gr.Textbox(label="Your message", submit_btn=True)
    clear = gr.Button("Clear")

    msg.submit(answer, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ("", []), None, [msg, chatbot])


if __name__ == "__main__":
    demo.launch()
