# AI Practice — Tool Calling vs MCP

This project demonstrates two approaches to giving an LLM the ability to call tools:

## Project Structure

```
AI_practice/
├── direct_tool_calling/       # Approach A: Everything in one process
│   ├── tools.py               # Tool definitions (schemas + implementations)
│   └── chatbot.py             # Gradio chatbot + Ollama + tool calling loop
│
├── mcp_approach/              # Approach B: Tools on a separate MCP server
│   ├── mcp_server.py          # MCP server (tools only, no LLM)
│   └── mcp_client_chatbot.py  # Gradio chatbot + Ollama + MCP client
│
├── requirements.txt
└── README.md
```

## Prerequisites

1. **Ollama** installed and running:
   ```bash
   ollama serve
   ollama pull qwen2.5:3b
   ```

2. **Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running Approach A: Direct Tool Calling

Everything in one process — tools are defined and executed locally.

```bash
cd direct_tool_calling
python chatbot.py
```

Open the Gradio URL (usually http://127.0.0.1:7860) and try:
- "What's the weather in Hyderabad?"
- "What time is it?"
- "Calculate (15 * 3) + 27"

## Running Approach B: MCP

Tools live on a separate MCP server. The client discovers and calls them via MCP protocol.

```bash
cd mcp_approach
python mcp_client_chatbot.py
```

> Note: The MCP server is started automatically by the `fastmcp` Client.
> You do NOT need to start it manually.

Open the Gradio URL and try the same prompts. The behavior is identical —
but internally, tool calls go through the MCP protocol to a separate server process.

## Key Differences

| Aspect | Approach A (Direct) | Approach B (MCP) |
|--------|-------------------|------------------|
| Tool location | Same process | Separate MCP server |
| Tool schemas | Hardcoded JSON | Auto-generated from function signatures |
| Tool discovery | Manual | Automatic (`tools/list`) |
| Tool execution | Direct function call | Via MCP protocol (`tools/call`) |
| Reusability | Tools tied to this app | Any MCP client can use the tools |
| Setup complexity | Simpler | Slightly more setup |

## What to Look For

1. **Console output** — Both approaches print detailed logs showing:
   - When the LLM decides to call a tool
   - What arguments it passes
   - What the tool returns
   - How the result is fed back to the LLM

2. **Compare the code** — Open `direct_tool_calling/chatbot.py` and
   `mcp_approach/mcp_client_chatbot.py` side by side. The chat loop is
   nearly identical — the only difference is WHERE tools come from.

3. **Try the same prompts** in both approaches to see identical behavior
   with different architectures.
