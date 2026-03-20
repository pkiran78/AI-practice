"""
MCP Client Chatbot — Gradio UI + Ollama + MCP Tools

This is APPROACH B (client side): Connects to the MCP server to discover and call tools.
The LLM (Ollama) lives HERE in the client. The tools live in the MCP server.

KEY DIFFERENCE FROM DIRECT TOOL CALLING:
- Tools are NOT defined here — they come from the MCP server
- Tool schemas are FETCHED at startup, not hardcoded
- Tool execution happens via MCP protocol, not direct function calls
- If the MCP server adds new tools, this client sees them automatically

HOW IT WORKS:
1. Connect to MCP server → discover tools (tools/list)
2. Convert MCP tool schemas to OpenAI format
3. User sends message → LLM decides to call a tool
4. Client sends tool call to MCP server (tools/call)
5. MCP server executes and returns result
6. Result goes back to LLM → LLM generates final response

RUN:
  First start the MCP server:  python mcp_server.py  (in another terminal)
  Then run this client:        python mcp_client_chatbot.py
"""

import asyncio
import json

import gradio as gr
from openai import OpenAI
from fastmcp import Client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"

# Path to the MCP server script — fastmcp Client can start it automatically
MCP_SERVER_PATH = "mcp_server.py"

llm_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
)

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
When the user asks something that requires a tool (weather, time, math), use the appropriate tool.
When the user asks a general question, just answer directly without using tools.
Always be concise and helpful."""


# ---------------------------------------------------------------------------
# MCP Tool Discovery — Fetch schemas from MCP server and convert to OpenAI format
# ---------------------------------------------------------------------------
def mcp_schema_to_openai(mcp_tool) -> dict:
    """
    Convert an MCP tool schema to OpenAI function-calling format.

    MCP format:
        name, description, inputSchema (JSON Schema)

    OpenAI format:
        {"type": "function", "function": {"name", "description", "parameters"}}
    """
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema if mcp_tool.inputSchema else {
                "type": "object",
                "properties": {},
            },
        }
    }


async def discover_tools(mcp_client: Client) -> list:
    """Connect to MCP server and fetch all available tool schemas."""
    mcp_tools = await mcp_client.list_tools()
    print(f"Discovered {len(mcp_tools)} tools from MCP server:")
    for t in mcp_tools:
        print(f"  - {t.name}: {t.description}")
    return mcp_tools


async def call_mcp_tool(mcp_client: Client, tool_name: str, arguments: dict) -> str:
    """Call a tool on the MCP server and return the result."""
    print(f"  MCP call: {tool_name}({json.dumps(arguments)})")
    result = await mcp_client.call_tool(tool_name, arguments)
    # FastMCP returns a CallToolResult object with a .content list
    if hasattr(result, 'content') and result.content:
        # Each content item may have a .text attribute
        parts = []
        for item in result.content:
            if hasattr(item, 'text'):
                parts.append(item.text)
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    else:
        text = str(result)
    print(f"  MCP result: {text[:200]}")
    return text


# ---------------------------------------------------------------------------
# Chat function with MCP tool calling loop
# ---------------------------------------------------------------------------
async def chat_with_mcp_tools(user_message: str, history: list, mcp_client: Client) -> str:
    """
    Same ReAct loop as direct tool calling, but tools are called via MCP.

    Compare with direct_tool_calling/chatbot.py — the structure is identical!
    The only difference: tool execution goes through MCP instead of local function calls.
    """

    # Step 0: Discover tools from MCP server (converted to OpenAI format)
    mcp_tools = await discover_tools(mcp_client)
    openai_tools = [mcp_schema_to_openai(t) for t in mcp_tools]

    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for entry in history:
        if entry["role"] in ("user", "assistant"):
            messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": user_message})

    # --- THE TOOL CALLING LOOP (same pattern as direct approach) ---
    max_iterations = 5
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        try:
            response = llm_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=openai_tools,  # <-- schemas FETCHED from MCP, not hardcoded
                temperature=0.3,
            )
        except Exception as e:
            return f"Error calling Ollama: {e}"

        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            print(f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")

            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            for tool_call in assistant_message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # KEY DIFFERENCE: Call tool via MCP server, not locally!
                result = await call_mcp_tool(mcp_client, fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            continue

        return assistant_message.content or "No response from LLM."

    return "Error: Too many tool call iterations."


# ---------------------------------------------------------------------------
# Gradio Chat UI with async MCP client
# ---------------------------------------------------------------------------
def create_app():
    """Create the Gradio app with MCP client integration."""

    # We'll manage the MCP client connection within the chat handler
    async def respond_async(message, chat_history):
        # Connect to MCP server for each request
        # (In production, you'd keep a persistent connection)
        async with Client(MCP_SERVER_PATH) as mcp_client:
            bot_response = await chat_with_mcp_tools(message, chat_history, mcp_client)

        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_response})
        return "", chat_history

    def respond(message, chat_history):
        return asyncio.run(respond_async(message, chat_history))

    with gr.Blocks(title="MCP Client Chatbot") as demo:
        gr.Markdown("""
        # MCP Client Chatbot
        ### Approach B: LLM here (client) + Tools on MCP server (separate process)

        **How this differs from Approach A:**
        - Tools are NOT defined in this file
        - Tool schemas are FETCHED from the MCP server at runtime
        - Tool execution happens via MCP protocol
        - The MCP server can be running anywhere

        **Available tools (discovered from MCP server):**
        - **get_weather** — Ask about weather in any city
        - **get_current_time** — Ask for current time
        - **calculate** — Ask it to do math

        **Try the same prompts as Approach A to compare!**
        """)

        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Your message", placeholder="Ask me anything...", submit_btn=True)
        clear = gr.Button("Clear Chat")

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: (None, []), None, [msg, chatbot])

    return demo


if __name__ == "__main__":
    print("Starting MCP Client Chatbot...")
    print(f"Using Ollama at {OLLAMA_BASE_URL} with model {OLLAMA_MODEL}")
    print(f"MCP Server: {MCP_SERVER_PATH}")
    print("The MCP server will be started automatically by fastmcp Client.\n")
    app = create_app()
    app.launch()
