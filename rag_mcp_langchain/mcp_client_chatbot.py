import asyncio
import json
import os

import gradio as gr
from fastmcp import Client
from langchain_openai import ChatOpenAI


HERE = os.path.dirname(__file__)
MCP_SERVER_PATH = "mcp_retriever_server.py"

OLLAMA_OPENAI_BASE_URL = os.environ.get("OLLAMA_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

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


def _should_use_retrieval(message: str) -> bool:
    text = (message or "").lower()
    if not text.strip():
        return False

    smalltalk = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "thanks",
        "thank you",
        "bye",
    }
    if any(p in text for p in smalltalk) and len(text) <= 30:
        return False

    cricket_keywords = {
        "cricket",
        "no ball",
        "no-ball",
        "wide",
        "wicket",
        "over",
        "innings",
        "powerplay",
        "t20",
        "odi",
        "test match",
        "batsman",
        "batter",
        "bowler",
        "run rate",
        "scorecard",
        "super over",
        "lbw",
        "duck",
        "six",
        "four",
        "boundary",
    }
    if any(k in text for k in cricket_keywords):
        return True

    question_triggers = ("?", "what", "why", "how", "explain", "define", "difference", "give me")
    if any(t in text for t in question_triggers):
        return True

    return False


def _build_context(hits: list[dict]) -> tuple[str, list[str]]:
    blocks = []
    sources = []
    for h in hits:
        doc_id = h.get("doc_id", "")
        snippet = h.get("snippet", "")
        if doc_id:
            sources.append(doc_id)
        blocks.append(f"[source: {doc_id}]\n{snippet}")
    return "\n\n".join(blocks), sources


def _format_sources(sources: list[str]) -> str:
    seen = set()
    ordered = []
    for s in sources:
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
    return "\n".join(f"- {s}" for s in ordered) if ordered else "- (no sources)"


async def rag_answer(message: str, history: list) -> tuple[str, list]:
    use_retrieval = _should_use_retrieval(message)

    if use_retrieval:
        async with Client(MCP_SERVER_PATH) as mcp_client:
            tool_result_text = await _call_mcp_search(mcp_client, message)

        parsed = json.loads(tool_result_text)
        hits = parsed.get("hits", []) if isinstance(parsed, dict) else []
        context, sources = _build_context(hits)

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"User question: {message}"
        )

        resp = LLM.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        text += "\n\nMode:\n- Retriever + LLM"
        text += "\n\nSources:\n" + _format_sources(sources)
    else:
        prompt = (
            "You are a helpful assistant. Answer conversationally. "
            "If the user asks about cricket facts, ask them to click 'Build/Rebuild Index' and then ask the question again.\n\n"
            f"User message: {message}"
        )

        resp = LLM.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        text += "\n\nMode:\n- LLM only (no retrieval)"

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": text})

    return "", history


async def build_index(history: list) -> tuple[str, list]:
    async with Client(MCP_SERVER_PATH) as mcp_client:
        result = await mcp_client.call_tool("index_cricket_docs", {"reset": True})

    if hasattr(result, "content") and result.content:
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    else:
        text = str(result)

    history = history or []
    history.append(
        {
            "role": "assistant",
            "content": f"Index build result:\n{text}",
        }
    )
    return "", history


async def _call_mcp_search(mcp_client: Client, query: str) -> str:
    result = await mcp_client.call_tool("search", {"query": query, "top_k": 5})

    # FastMCP returns a CallToolResult-like object with .content list
    if hasattr(result, "content") and result.content:
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(result)


def respond(message, chat_history):
    return asyncio.run(rag_answer(message, chat_history))


def rebuild_index(chat_history):
    return asyncio.run(build_index(chat_history))


with gr.Blocks(title="RAG via MCP (LangChain + Chroma)") as demo:
    gr.Markdown(
        "# RAG via MCP (LangChain + Chroma)\n"
        "Retriever runs in an MCP server; client does LLM + UI."
    )

    chatbot = gr.Chatbot(height=520)
    msg = gr.Textbox(label="Your message", submit_btn=True)
    rebuild = gr.Button("Build/Rebuild Index")
    clear = gr.Button("Clear")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    rebuild.click(rebuild_index, [chatbot], [msg, chatbot])
    clear.click(lambda: ("", []), None, [msg, chatbot])


if __name__ == "__main__":
    demo.launch()
