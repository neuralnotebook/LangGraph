# Model Context Protocol (MCP) — Complete Notes

---

## 1. What is MCP?

**Model Context Protocol (MCP)** is an **open-source protocol** introduced by Anthropic that standardizes how applications provide context to Large Language Models (LLMs).

> **Analogy:** Think of MCP like a **USB-C port** for AI applications. Just as a USB-C port lets you connect many different devices (hard drives, cameras, phones) through one universal interface, MCP lets LLMs connect to many different tools and services through one universal protocol.

---

## 2. Why Do We Need MCP?

### The Problem with Traditional LLM Tool Integration

Before MCP, when you wanted to give an LLM access to external tools (e.g., Wikipedia, Arxiv, databases, web search), you had to:

1. Write **custom integration code** for each tool
2. Import the tool's specific libraries
3. Write wrapper code and glue logic
4. **Maintain** that code yourself — if the tool provider updated their API, you had to update your integration too

```
LLM Assistant
    ├── custom code → Arxiv tool
    ├── custom code → Wikipedia tool
    ├── custom code → RAG Database
    └── custom code → DuckDuckGo Search
```

**Scaling problem:** If your LLM needs 100 tools, you write and manage 100 custom integrations. Any update from any provider breaks your code.

### How MCP Solves This

MCP introduces a **common protocol layer** between LLMs and tool providers:

```
LLM Assistant
    └── MCP Protocol → MCP Server (managed by tool provider)
                            ├── Arxiv
                            ├── Wikipedia
                            ├── Database
                            └── Web Search
```

**Key insight:** The MCP Server and the tools it exposes are **maintained by the service provider**, not by you. When the provider updates their service, they update their MCP server. Your integration code stays the same — because the protocol itself doesn't change.

---

## 3. Understanding the Protocol (Analogy: HTTP / REST APIs)

Before diving into MCP components, it helps to think of how the web works:

- A **client** (browser) communicates with a **server** (website backend) using the **HTTP protocol**
- HTTP is a common language. Whether you're loading a webpage (`GET`) or submitting a form (`POST`), you follow the same standard
- REST APIs expose backend services using JSON over HTTP — a standard way for any client to talk to any server

**MCP works the same way**, but for AI tools. It is the "REST API" of the LLM tool ecosystem.

---

## 4. Core Components of MCP

There are **three main components**:

### 4.1 MCP Host

The application or environment that *uses* MCP. It:
- Hosts the entire MCP setup
- Creates MCP Clients internally
- Displays results to the user

**Examples of MCP Hosts:**
- Cursor IDE
- VS Code
- Windsurf IDE
- Claude Desktop
- Your own custom app (built with Streamlit, FastAPI, etc.)

### 4.2 MCP Client

- Lives **inside** the MCP Host
- Maintains a **one-to-one connection** with an MCP Server
- Communicates using the MCP protocol

### 4.3 MCP Server

- The bridge between your LLM/client and the actual tool or service
- Exposes **tools, context, and prompts** to the client
- Connected to real resources: databases, APIs, code repos, etc.
- **Managed by the service provider** — not by you

```
┌──────────────────────────────────────────────┐
│               MCP Host (e.g. Cursor IDE)      │
│                                              │
│   ┌────────────┐      ┌────────────────────┐ │
│   │ MCP Client │─────▶│    LLM / AI Agent  │ │
│   └─────┬──────┘      └────────────────────┘ │
└─────────┼────────────────────────────────────┘
          │ MCP Protocol
          ▼
┌─────────────────────┐    ┌─────────────────────┐
│    MCP Server #1    │    │    MCP Server #2     │
│  (Math Tools)       │    │  (Weather API)       │
│  - add()            │    │  - get_weather()     │
│  - multiply()       │    │                     │
└─────────────────────┘    └─────────────────────┘
```

---

## 5. How the Communication Flow Works

Step-by-step flow when a user gives input:

```
1. User provides input to the MCP Host (e.g., "What's 3 + 5 * 2?")

2. MCP Host queries MCP Servers → gets list of available tools

3. MCP Host sends to LLM:
      { question: "What's 3 + 5 * 2?", tools: [add, multiply, get_weather] }

4. LLM decides which tool to use → responds: "Use the 'add' then 'multiply' tool"

5. MCP Host calls that specific tool through the MCP Server

6. Tool returns result → context is sent back to LLM

7. LLM generates final response → displayed to user
```

---

## 6. Transport Mechanisms

When MCP Client communicates with MCP Server, it uses a **transport protocol** — the communication channel.

### 6.1 stdio (Standard Input/Output)

- The server runs as a **local process** on the command line
- Client communicates via stdin/stdout (the terminal)
- Useful for **local development and testing**
- Does NOT run as an HTTP server — it just reads/writes from the terminal

```python
# Server uses stdio transport
mcp.run(transport="stdio")
```

When you run this server (`python math_server.py`), it does NOT start a web server. Instead, it reads from stdin and writes to stdout — perfect for local tool testing.

### 6.2 Streamable HTTP

- The server runs as a **full HTTP API service**
- Accessible via a URL (e.g., `http://localhost:8000/mcp`)
- Suitable for **production deployments**, remote servers, third-party integrations
- The client communicates over HTTP

```python
# Server uses streamable HTTP transport
mcp.run(transport="streamable-http")
```

When you run this (`python weather.py`), it starts a proper HTTP server at `localhost:8000`.

| Feature          | stdio                    | Streamable HTTP           |
|------------------|--------------------------|---------------------------|
| Runs as          | Local process (terminal) | HTTP API (web server)     |
| Best for         | Local dev & testing      | Production / remote use   |
| Communication    | stdin / stdout           | HTTP requests             |
| URL needed?      | No                       | Yes (`/mcp` endpoint)     |

---

## 7. Building MCP: Complete Code Walkthrough

### Project Setup (using `uv` package manager)

```bash
# Initialize project
uv init

# Create virtual environment
uv venv

# Activate virtual environment (Mac/Linux)
source .venv/bin/activate
```

### requirements.txt

```
langchain-groq
langchain-mcp-adapters
fastmcp
langgraph
python-dotenv
```

```bash
# Install all dependencies
uv add -r requirements.txt
```

---

### 7.1 MCP Server — Math Server (`math_server.py`)

This server exposes two tools: `add` and `multiply`. It uses **stdio transport**.

```python
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server with a name
mcp = FastMCP("math")

# Define tool using @mcp.tool() decorator
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    # transport = "stdio"
    # Tells the server to use standard input/output to receive
    # and respond to tool function calls.
    # This runs the server as a local process (not a web server).
    # Client communicates with it via command-line stdin/stdout.
    mcp.run(transport="stdio")
```

**Key points:**
- `FastMCP("math")` — creates a server with the name "math"
- `@mcp.tool()` — decorator that registers a function as an MCP tool
- The **docstring** (`"""Add two numbers"""`) is critical — the LLM reads this to decide which tool to call
- `transport="stdio"` — runs as a local process, not a web server

---

### 7.2 MCP Server — Weather Server (`weather.py`)

This server exposes a weather tool. It uses **Streamable HTTP transport**.

```python
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("weather")

@mcp.tool()
def get_weather(location: str) -> str:
    """Get the weather for a given location"""
    # In a real scenario, this would call a weather API
    # e.g., requests.get(f"https://api.weather.com/{location}")
    # For demo purposes, returning a hardcoded response:
    return "It's always raining in California."

if __name__ == "__main__":
    # transport = "streamable-http"
    # Runs the MCP server as a proper HTTP web service.
    # Default URL: http://localhost:8000
    # Tools are accessible at: http://localhost:8000/mcp
    # Use this for production or when you want a persistent server.
    mcp.run(transport="streamable-http")
```

**Run this server:**
```bash
python weather.py
# Output: Uvicorn running on http://localhost:8000
```

Now the server is live at `http://localhost:8000/mcp`.

---

### 7.3 Environment File (`.env`)

```
GROQ_API_KEY=your_groq_api_key_here
```

---

### 7.4 MCP Client — Connecting to Both Servers (`client.py`)

This is the main application. It:
- Creates a **MultiServerMCPClient** that connects to both servers
- Loads all available tools from both servers
- Creates a **LangGraph ReAct Agent** with those tools + an LLM
- Invokes the agent with user questions

```python
import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

async def main():
    # ─────────────────────────────────────────────────────────────────
    # Step 1: Create the Multi-Server MCP Client
    # This client maintains connections to all your MCP servers.
    # ─────────────────────────────────────────────────────────────────
    client = MultiServerMCPClient(
        {
            # Server 1: Math server — uses stdio transport
            # "command" tells the client how to launch this server
            # It will run: python /absolute/path/to/math_server.py
            "math": {
                "command": "python",
                "args": ["math_server.py"],  # ensure correct absolute path
                "transport": "stdio",
            },

            # Server 2: Weather server — uses streamable HTTP transport
            # The server must already be RUNNING before the client connects
            # Run: python weather.py  (in a separate terminal)
            "weather": {
                "url": "http://localhost:8000/mcp",  # ensure server is running
                "transport": "streamable_http",
            },
        }
    )

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Load all tools from all connected MCP servers
    # client.get_tools() returns a unified list of tools from
    # both the math server (add, multiply) and weather server (get_weather)
    # ─────────────────────────────────────────────────────────────────
    tools = await client.get_tools()

    # ─────────────────────────────────────────────────────────────────
    # Step 3: Initialize the LLM
    # ─────────────────────────────────────────────────────────────────
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    model = ChatGroq(model="qwen-qwq-32b")

    # ─────────────────────────────────────────────────────────────────
    # Step 4: Create a ReAct Agent
    # This agent:
    #   - Takes user input
    #   - Decides which tool(s) to call based on the question
    #   - Calls the tool via MCP protocol
    #   - Returns the final answer
    # ─────────────────────────────────────────────────────────────────
    agent = create_react_agent(model, tools)

    # ─────────────────────────────────────────────────────────────────
    # Step 5: Invoke the agent with a math question
    # The LLM will figure out it needs to call 'add' then 'multiply'
    # ─────────────────────────────────────────────────────────────────
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 3 + 5 * 2?"}]}
    )
    print("Math Response:", math_response["messages"][-1].content)

    # ─────────────────────────────────────────────────────────────────
    # Step 6: Invoke the agent with a weather question
    # The LLM will figure out it needs to call 'get_weather'
    # ─────────────────────────────────────────────────────────────────
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the weather in California?"}]}
    )
    print("Weather Response:", weather_response["messages"][-1].content)


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
```

**Run the client:**
```bash
# In terminal 1: start the weather HTTP server
python weather.py

# In terminal 2: run the client
python client.py
```

**Expected output:**
```
Math Response: The result of 3 + 5 * 2 is 13.
  Step 1: 3 + 5 = 8 (add tool called)
  Step 2: 8 * 2 = 16 ... (multiply tool called)

Weather Response: According to the tool, it's always raining in California.
  (In reality, California has a diverse climate...)
```

---

## 8. Project File Structure

```
mcp-demo/
├── .env                  # API keys
├── .venv/                # Virtual environment
├── pyproject.toml        # Project metadata (auto-generated by uv)
├── requirements.txt      # Dependencies
├── math_server.py        # MCP Server #1 (stdio transport)
├── weather.py            # MCP Server #2 (streamable HTTP transport)
└── client.py             # MCP Client + LangGraph Agent
```

---

## 9. Key Libraries Used

| Library | Purpose |
|---|---|
| `fastmcp` | Easy Pythonic way to build MCP servers and tools |
| `langchain-mcp-adapters` | Bridges MCP tools into LangChain/LangGraph |
| `langgraph` | Creates ReAct agents that can use MCP tools |
| `langchain-groq` | LLM provider (Groq's fast inference) |
| `python-dotenv` | Loads `.env` files for API keys |

---

## 10. The @mcp.tool() Decorator — Why Docstrings Matter

```python
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""   # ← LLM reads THIS to decide when to use the tool
    return a + b
```

The **docstring is not optional** — it's the tool's description that gets sent to the LLM. The LLM uses it to decide:
- Does this tool match the user's intent?
- Should I call this tool now?

Write clear, descriptive docstrings. For a real weather API:
```python
@mcp.tool()
def get_weather(location: str) -> str:
    """Get current weather conditions for a given city or location.
    Returns temperature, humidity, and conditions."""
    # ... real API call here
```

---

## 11. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Input                                                     │
│      │                                                          │
│      ▼                                                          │
│  MCP HOST (Cursor / Claude Desktop / Your App)                  │
│      │                                                          │
│      ├── MCP CLIENT ──[MCP Protocol]──▶ MCP SERVER (stdio)     │
│      │                                      └── add()          │
│      │                                      └── multiply()     │
│      │                                                          │
│      └── MCP CLIENT ──[MCP Protocol]──▶ MCP SERVER (HTTP)     │
│                                              └── get_weather() │
│                                                                  │
│  LLM decides which tool → tool called → context returned → answer│
└─────────────────────────────────────────────────────────────────┘
```

**Key Takeaways:**
1. MCP is a **universal protocol** — like REST API but for LLM tools
2. Three components: **Host → Client → Server**
3. Two transport modes: **stdio** (local/dev) and **streamable HTTP** (production)
4. Service providers **manage their own MCP servers** — you just connect
5. Use `FastMCP` to build servers, `MultiServerMCPClient` + LangGraph to build clients
6. **Docstrings** on tools are read by the LLM to decide which tool to call