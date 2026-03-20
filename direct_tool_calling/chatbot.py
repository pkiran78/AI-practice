"""
Direct Tool Calling Chatbot — Gradio UI + Ollama + Tools

This is APPROACH A: Everything in one process.
- Tools are defined in tools.py (schemas + implementations)
- The LLM (Ollama) decides which tool to call
- This code executes the tool and feeds the result back to the LLM
- Gradio provides the chat UI

HOW IT WORKS:
1. User sends a message
2. We send it to Ollama along with tool schemas
3. Ollama may respond with text OR a tool_call
4. If tool_call → we execute the function → send result back to Ollama
5. Ollama generates a final text response
6. Display to user

RUN: python chatbot.py
"""

import json
from openai import OpenAI
import gradio as gr

from tools import TOOLS, TOOL_MAP

# ---------------------------------------------------------------------------
# Configuration — Ollama running locally
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"  # Make sure this model is pulled in Ollama

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",  # Ollama doesn't need a real key
)

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
When the user asks something that requires a tool (weather, time, math), use the appropriate tool.
When the user asks a general question, just answer directly without using tools.
Always be concise and helpful."""


# ---------------------------------------------------------------------------
# Core chat function with tool calling loop
# ---------------------------------------------------------------------------
def chat_with_tools(user_message: str, history: list) -> tuple:
    """
    Process a user message, potentially calling tools, and return the response.

    The tool calling loop:
    1. Send messages + tool schemas to LLM
    2. If LLM returns tool_calls → execute them → add results → go to step 1
    3. If LLM returns text → return it
    """

    # Build messages from history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Convert Gradio history format to OpenAI format
    for entry in history:
        if entry["role"] == "user":
            messages.append({"role": "user", "content": entry["content"]})
        elif entry["role"] == "assistant":
            messages.append({"role": "assistant", "content": entry["content"]})

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # --- THE TOOL CALLING LOOP ---
    max_iterations = 5  # safety limit to prevent infinite loops
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"Sending {len(messages)} messages to LLM with {len(TOOLS)} tools")

        # Step 1: Call the LLM with tools
        try:
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=0.3,
            )
        except Exception as e:
            return f"Error calling Ollama: {e}\n\nMake sure Ollama is running (`ollama serve`) and model `{OLLAMA_MODEL}` is pulled.", history

        assistant_message = response.choices[0].message

        # Step 2: Check if LLM wants to call tools
        if assistant_message.tool_calls:
            print(f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")

            # Add the assistant's message (with tool calls) to history
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

            # Step 3: Execute each tool call
            for tool_call in assistant_message.tool_calls:
                fn_name = tool_call.function.name
                fn_args_str = tool_call.function.arguments

                print(f"  Calling tool: {fn_name}({fn_args_str})")

                # Parse arguments
                try:
                    fn_args = json.loads(fn_args_str)
                except json.JSONDecodeError:
                    fn_args = {}

                # Execute the tool
                if fn_name in TOOL_MAP:
                    result = TOOL_MAP[fn_name](**fn_args)
                    print(f"  Tool result: {result}")
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"})

                # Step 4: Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Loop back to Step 1 — LLM will now see the tool results
            continue

        # Step 5: No tool calls — LLM gave a text response
        final_text = assistant_message.content or "No response from LLM."
        print(f"LLM final response: {final_text[:100]}...")
        return final_text, history

    return "Error: Too many tool call iterations.", history


# ---------------------------------------------------------------------------
# Gradio Chat UI
# ---------------------------------------------------------------------------
def respond(message, chat_history):
    """Gradio chat handler."""
    bot_response, _ = chat_with_tools(message, chat_history)
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_response})
    return "", chat_history


with gr.Blocks(title="Direct Tool Calling Chatbot") as demo:
    gr.Markdown("""
    # Direct Tool Calling Chatbot
    ### Approach A: LLM + Tools in one process

    **Available tools:**
    - **get_weather** — Ask about weather in any city
    - **get_current_time** — Ask for current time
    - **calculate** — Ask it to do math

    **Try these:**
    - "What's the weather in Hyderabad?"
    - "What time is it?"
    - "Calculate (15 * 3) + 27"
    - "What's 2+2 and what's the weather in London?" (multi-tool!)
    """)

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your message", placeholder="Ask me anything...", submit_btn=True)
    clear = gr.Button("Clear Chat")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: (None, []), None, [msg, chatbot])

if __name__ == "__main__":
    print("Starting Direct Tool Calling Chatbot...")
    print(f"Using Ollama at {OLLAMA_BASE_URL} with model {OLLAMA_MODEL}")
    print("Make sure Ollama is running: `ollama serve`")
    print(f"Make sure model is pulled: `ollama pull {OLLAMA_MODEL}`\n")
    demo.launch()
