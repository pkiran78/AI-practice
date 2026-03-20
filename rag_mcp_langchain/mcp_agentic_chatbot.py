import asyncio
import json
import os

import gradio as gr
from fastmcp import Client
from openai import OpenAI


OLLAMA_OPENAI_BASE_URL = os.environ.get("OLLAMA_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

MCP_SERVER_PATH = "mcp_retriever_server.py"

llm_client = OpenAI(
    base_url=OLLAMA_OPENAI_BASE_URL,
    api_key="ollama",
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. You may call tools when needed. "
    "When answering cricket questions, prefer using the retriever tool for grounding and cite sources as [source: <doc_id>]."
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


async def _discover_openai_tools(mcp_client: Client) -> list[dict]:
    mcp_tools = await mcp_client.list_tools()
    return [mcp_schema_to_openai(t) for t in mcp_tools]


async def _call_mcp_tool(mcp_client: Client, tool_name: str, arguments: dict) -> str:
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


def _history_to_messages(history: list) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in history or []:
        if isinstance(entry, dict) and entry.get("role") in ("user", "assistant", "tool"):
            m = {"role": entry.get("role"), "content": entry.get("content", "")}
            if entry.get("role") == "tool" and entry.get("tool_call_id"):
                m["tool_call_id"] = entry["tool_call_id"]
            messages.append(m)
    return messages


async def agent_turn(user_message: str, history: list) -> tuple[str, list]:
    history = history or []
    history.append({"role": "user", "content": user_message})

    max_tool_calls = 3
    async with Client(MCP_SERVER_PATH) as mcp_client:
        openai_tools = await _discover_openai_tools(mcp_client)

        for _ in range(max_tool_calls):
            messages = _history_to_messages(history)

            resp = llm_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=openai_tools,
                temperature=0.2,
            )

            msg = resp.choices[0].message

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")

                    history.append(
                        {
                            "role": "assistant",
                            "content": f"Tool called: {name}({json.dumps(args, ensure_ascii=False)})",
                        }
                    )

                    result_text = await _call_mcp_tool(mcp_client, name, args)

                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result_text,
                        }
                    )

                continue

            final = msg.content or "No response from model."
            final += "\n\nMode:\n- Agentic (LLM chose tools)"
            history.append({"role": "assistant", "content": final})
            return "", history

    history.append(
        {
            "role": "assistant",
            "content": "Stopped after max tool calls (3). Please ask a simpler question or be more specific.\n\nMode:\n- Agentic (LLM chose tools)",
        }
    )
    return "", history


def respond(message, chat_history):
    return asyncio.run(agent_turn(message, chat_history))


with gr.Blocks(title="Agentic RAG via MCP (LangChain + Chroma)") as demo:
    gr.Markdown(
        "# Agentic Assistant (MCP + Multiple Tools)\n"
        "This demo lets the LLM decide whether to call tools. All tools are discovered dynamically from the MCP server."
    )

    chatbot = gr.Chatbot(height=520)
    msg = gr.Textbox(label="Your message", submit_btn=True)
    clear = gr.Button("Clear")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ("", []), None, [msg, chatbot])


if __name__ == "__main__":
    demo.launch()
