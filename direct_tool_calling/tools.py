"""
Direct Tool Calling — Tool Definitions & Implementations

This file defines:
1. Python functions that ARE the tools
2. JSON schemas that DESCRIBE the tools to the LLM
3. A mapping from tool name → function (so we can dispatch calls)

The LLM reads the schemas to decide which tool to call.
Your code uses the mapping to actually execute the tool.
"""

import json
import random
from datetime import datetime


# ---------------------------------------------------------------------------
# Tool 1: Get Weather (dummy — for learning)
# ---------------------------------------------------------------------------
def get_weather(city: str) -> str:
    """Simulate getting weather for a city."""
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
def get_current_time(timezone: str = "UTC") -> str:
    """Return the current date and time."""
    now = datetime.now()
    return json.dumps({
        "timezone": timezone,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "This is local server time (timezone param is illustrative).",
    })


# ---------------------------------------------------------------------------
# Tool 3: Calculate (simple math evaluator)
# ---------------------------------------------------------------------------
def calculate(expression: str) -> str:
    """Evaluate a simple math expression safely."""
    try:
        # Only allow safe math operations
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return json.dumps({"error": "Only basic math operators allowed"})
        result = eval(expression)  # safe because we filtered characters
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# TOOL SCHEMAS — This is what the LLM reads to decide which tool to call
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city. Returns temperature, condition, and humidity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'London', 'New York', 'Hyderabad'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time. Optionally specify a timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name like 'UTC', 'IST', 'EST'. Default is UTC."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Supports +, -, *, /, parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '(2 + 3) * 4'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
]

# ---------------------------------------------------------------------------
# TOOL MAP — Routes tool name → actual Python function
# ---------------------------------------------------------------------------
TOOL_MAP = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "calculate": calculate,
}
