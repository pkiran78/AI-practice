import asyncio
import json
import os

import gradio as gr
from fastmcp import Client
from openai import OpenAI


OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"

MCP_SERVER_PATH = "mcp_rag_server.py"
SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
CRICKET_DOCS_DIR = os.path.join(os.path.dirname(__file__), "cricket_docs")

llm_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about cricket using the provided context. "
    "Use only the provided context for factual claims, and cite sources as [source: <doc_id>]. "
    "If the context is insufficient, say you don't have enough information in the indexed docs."
)


def mcp_schema_to_openai(mcp_tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema
            if mcp_tool.inputSchema
            else {"type": "object", "properties": {}},
        },
    }


async def discover_tools(mcp_client: Client) -> list:
    return await mcp_client.list_tools()


async def call_mcp_tool(mcp_client: Client, tool_name: str, arguments: dict) -> str:
    result = await mcp_client.call_tool(tool_name, arguments)
    if hasattr(result, "content") and result.content:
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(result)


def _build_context(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        doc_id = h.get("doc_id", "")
        snippet = h.get("snippet", "")
        lines.append(f"[source: {doc_id}]\n{snippet}")
    return "\n\n".join(lines)


async def _ensure_indexed(mcp_client: Client) -> None:
    docs_dir = CRICKET_DOCS_DIR if os.path.isdir(CRICKET_DOCS_DIR) else SAMPLE_DOCS_DIR
    glob_pattern = "*.md" if docs_dir == CRICKET_DOCS_DIR else "*.txt"

    result_text = await call_mcp_tool(
        mcp_client,
        "index_directory",
        {"directory": docs_dir, "glob_pattern": glob_pattern},
    )

    try:
        parsed = json.loads(result_text)
    except Exception as e:
        raise RuntimeError(f"Indexing failed: could not parse MCP response: {result_text}") from e

    if isinstance(parsed, dict) and parsed.get("error"):
        raise RuntimeError(
            f"Indexing failed: {parsed['error']} (directory={docs_dir}, glob_pattern={glob_pattern})"
        )


async def retrieve_context(mcp_client: Client, query: str, top_k: int = 5) -> tuple[str, list[str]]:
    tool_result_text = await call_mcp_tool(
        mcp_client,
        "search",
        {"query": query, "top_k": top_k},
    )

    parsed = json.loads(tool_result_text)
    hits = parsed.get("hits", []) if isinstance(parsed, dict) else []
    context_block = _build_context(hits)
    sources = [h.get("doc_id", "") for h in hits if isinstance(h, dict) and h.get("doc_id")]
    return context_block, sources


async def chat_with_rag(user_message: str, history: list, mcp_client: Client) -> str:
    context_block, sources = await retrieve_context(mcp_client, user_message, top_k=5)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in history:
        if isinstance(entry, dict) and entry.get("role") in ("user", "assistant"):
            messages.append({"role": entry["role"], "content": entry.get("content", "")})

    messages.append(
        {
            "role": "user",
            "content": (
                "Answer the user question using the following retrieved context.\n\n"
                f"Retrieved context:\n{context_block}\n\n"
                f"User question: {user_message}"
            ),
        }
    )

    resp = llm_client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=messages,
        temperature=0.2,
    )

    answer = resp.choices[0].message.content or "No response from model."
    if sources:
        sources_line = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
    else:
        sources_line = "\n\nSources:\n- (no hits)"
    return answer + sources_line


def create_app():
    async def respond_async(message, chat_history):
        async with Client(MCP_SERVER_PATH) as mcp_client:
            try:
                await _ensure_indexed(mcp_client)
            except Exception as e:
                chat_history = chat_history or []
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": str(e)})
                return "", chat_history
            bot_response = await chat_with_rag(message, chat_history, mcp_client)

        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_response})
        return "", chat_history

    def respond(message, chat_history):
        return asyncio.run(respond_async(message, chat_history))

    with gr.Blocks(title="RAG via MCP (Learning Demo)") as demo:
        gr.Markdown(
            "# RAG via MCP\n"
            "This demo uses an MCP server as a retriever (index + search). "
            "The LLM runs locally via Ollama and calls MCP tools to fetch context."
        )

        chatbot = gr.Chatbot(height=520)
        msg = gr.Textbox(label="Your message", submit_btn=True)
        clear = gr.Button("Clear")

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: ("", []), None, [msg, chatbot])

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch()
