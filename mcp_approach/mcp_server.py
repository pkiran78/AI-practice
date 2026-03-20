"""
MCP Server — Tool Provider (using FastMCP)

This is APPROACH B (server side): Tools live here as a SEPARATE process.
Any MCP-compatible client can connect, discover these tools, and call them.

KEY DIFFERENCE FROM DIRECT TOOL CALLING:
- No LLM here — this is just a tool server
- No UI here — the client provides the UI
- No chat loop — the client handles conversation flow
- Tool schemas are AUTO-GENERATED from function signatures + docstrings

HOW IT WORKS:
1. FastMCP reads your @mcp.tool() decorated functions
2. It auto-generates JSON schemas from type hints + docstrings
3. When a client connects, it serves the tool list
4. When a client calls a tool, it executes the function and returns results

RUN: python mcp_server.py
  or: fastmcp run mcp_server.py
"""

import json
import random
from datetime import datetime

from fastmcp import FastMCP

# Create the MCP server with a name
mcp = FastMCP(
    "learning-tools-server",
    instructions="A learning MCP server with weather, time, and calculator tools.",
)


# ---------------------------------------------------------------------------
# Tool 1: Get Weather
# Notice: No JSON schema needed! FastMCP generates it from the function.
# ---------------------------------------------------------------------------
@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get the current weather for a given city.
    Returns temperature, condition, and humidity as JSON.

    Args:
        city: The city name, e.g. 'London', 'New York', 'Hyderabad'
    """
    weather_data = {
        "city": city,
        "temperature": random.randint(15, 35),
        "condition": random.choice(["sunny", "cloudy", "rainy", "windy"]),
        "humidity": random.randint(30, 90),
    }
    return json.dumps(weather_data)


# ---------------------------------------------------------------------------
# Tool 2: Get Current Time
# ---------------------------------------------------------------------------
@mcp.tool()
def get_live_cricket_score(team: str | None = None) -> str:
    """
    Get the cricket score

    Args:
        team: Cricket team name, e.g. 'India', 'Australia'
    """
    # timezone = timezone or "UTC"
    # now = datetime.now()
    return json.dumps({
        "team": team,
        "score": random.randint(100,200),
        "note": "This live score",
    })


# ---------------------------------------------------------------------------
# Tool 3: Calculate
# ---------------------------------------------------------------------------
@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression. Supports +, -, *, /, parentheses.

    Args:
        expression: Math expression to evaluate, e.g. '(2 + 3) * 4'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Only basic math operators allowed"})
        result = eval(expression)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Run the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting MCP Server: learning-tools-server")
    print("Tools available: get_weather, get_current_time, calculate")
    print("Waiting for MCP client connections...\n")
    mcp.run()
