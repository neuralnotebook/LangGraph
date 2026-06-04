# LangGraph — Complete Crash Course Notes (Part 1)

> Based on Krishna's LangGraph crash course. Covers project setup, core concepts (State, Nodes, Edges), building a basic chatbot, and integrating tools with conditional routing.

---

## 1. Project Setup

Same UV-based setup as LangChain, but with LangGraph-specific libraries.

### `requirements.txt`

```
langgraph
langchain
langsmith
langchain-groq
langchain-openai
python-dotenv
ipykernel
langchain-tavily
```

### Setup Commands

```bash
uv init                        # initialize project workspace
uv venv                        # create virtual environment (Python 3.13)
.venv\Scripts\activate         # activate (Windows)
source .venv/bin/activate      # activate (macOS/Linux)
uv add -r requirements.txt     # install all libraries
uv add ipykernel               # for Jupyter notebook support
```

### `.env` File

```env
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key   # get free from tavily.com
LANGSMITH_API_KEY=your_key       # for tracking/evaluation
```

### Load Environment in Code

```python
import os
from dotenv import load_dotenv

load_dotenv()
```

---

## 2. LangGraph Core Concepts

LangGraph solves **complex multi-step workflows** using a **graph structure** instead of a simple linear LLM call.

### The Three Components

| Component | What It Is | Analogy |
|---|---|---|
| **Node** | A unit of work — a Python function | A step in the workflow |
| **Edge** | A connection between nodes — defines flow of execution | An arrow between steps |
| **State** | A shared dictionary accessible by all nodes | Shared memory of the graph |

---

### 2.1 State

The **state** is a TypedDict class that stores variables shared across the entire graph. Any node can read from or write to it.

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    """
    'messages' is a list. The add_messages reducer appends to this list
    rather than overwriting it, preserving the full conversation history.
    """
    messages: Annotated[list, add_messages]
```

**Key concepts:**
- `TypedDict` — makes State return as a plain dictionary (key-value pairs)
- `Annotated[list, add_messages]` — attaches `add_messages` as a **reducer**
- **Reducer** — controls *how* a state variable gets updated. `add_messages` appends instead of replacing

> Without `add_messages`, each new message would **overwrite** the previous one. With it, every message gets **appended** to the list — maintaining full conversation history.

---

### 2.2 Nodes

A **node** is just a Python function that:
- Takes `state: State` as input
- Returns a dictionary with updated state values

```python
def chatbot(state: State):
    """
    Node definition for the chatbot.
    Takes messages from state, invokes the LLM, returns updated messages.
    """
    return {"messages": llm.invoke(state["messages"])}
```

---

### 2.3 Edges

An **edge** defines the direction of execution between nodes.

```python
# Simple edge: A → B (always goes from A to B)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Conditional edge: A → B or A → C depending on a condition
builder.add_conditional_edges("chatbot", tools_condition)
```

---

### 2.4 State Graph

The **StateGraph** is the container for the entire graph — all nodes, edges, and the state schema.

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
```

---

## 3. Building a Basic Chatbot

**Graph structure:**
```
START → [chatbot node] → END
```

### Full Code

```python
# ── Imports ────────────────────────────────────────────────────────────────
import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

load_dotenv()

# ── 1. Define the LLM ──────────────────────────────────────────────────────

# Option A — provider-specific
llm = ChatGroq(model="llama3-8b-8192")

# Option B — generic (recommended)
llm = init_chat_model("groq:llama3-8b-8192")

# ── 2. Define the State ────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── 3. Define the Node ────────────────────────────────────────────────────
def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}

# ── 4. Build the Graph ────────────────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("llm_chatbot", chatbot)     # node name + node function
builder.add_edge(START, "llm_chatbot")       # START → chatbot
builder.add_edge("llm_chatbot", END)         # chatbot → END

# ── 5. Compile ────────────────────────────────────────────────────────────
graph = builder.compile()
```

### Visualize the Graph

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass
# Output: START ──→ llm_chatbot ──→ END
```

### Run the Graph — `invoke`

```python
# Provide input via the 'messages' key (matches the State variable)
response = graph.invoke({"messages": "Hi"})

# Read the last message content
print(response["messages"][-1].content)
# → "Hi! It's nice to meet you. Is there something I can help you with?"
```

> When you pass a plain string like `"Hi"`, LangGraph automatically converts it to a `HumanMessage`. The LLM's response is an `AIMessage`. Both get appended to `state["messages"]` via the `add_messages` reducer.

### Run the Graph — `stream`

```python
# Stream events from the graph
for event in graph.stream({"messages": "Hi, how are you?"}):
    for value in event.values():
        print(value["messages"][-1].content)
# Output streams progressively as nodes execute
```

---

## 4. Chatbot with External Tools

When the LLM doesn't have the required information (e.g. live news, real-time data), it needs to call **external tools**.

**Graph structure:**
```
START → [tool_calling_llm] ──(tool call?)──→ [tools node] → END
                            └──(no tool call)──────────────→ END
```

### 4.1 Why Tools?

LLMs have a **training cutoff** — they can't answer questions about today's news or live data. By binding tools to the LLM:
- The LLM uses docstrings to understand what each tool does
- When input matches a tool's purpose → it makes a **tool call**
- The tool executes and returns context → LLM generates final answer

---

### 4.2 Setting Up Tavily (Web Search Tool)

Tavily is a real-time web search API built for LLMs. Free tier available at [tavily.com](https://tavily.com).

```python
from langchain_tavily import TavilySearch

# Initialize the search tool
tavily_tool = TavilySearch(max_results=2)

# Test it directly
result = tavily_tool.invoke("What is LangGraph?")
print(result)   # returns title, content, URL from live web
```

---

### 4.3 Creating a Custom Tool

```python
def multiply(a: int, b: int) -> int:
    """
    Multiply a and b.

    Args:
        a: First integer
        b: Second integer

    Returns:
        int: The product of a and b
    """
    return a * b
```

> The **docstring** is the schema. The LLM reads it to decide when to call this tool. Always write a clear, descriptive docstring.

---

### 4.4 Binding Tools to the LLM

```python
tools = [tavily_tool, multiply]

llm_with_tools = llm.bind_tools(tools)
# → Returns a RunnableBinding: ChatGroq + tool schemas attached
```

When `bind_tools` is called:
- The LLM now **knows** about these tools
- On each invocation it decides: "Can I answer this? Or do I need a tool?"

---

### 4.5 Tool Node

`ToolNode` from `langgraph.prebuilt` automatically executes whichever tool the LLM decided to call.

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)  # pass the same tools list
```

---

### 4.6 Tools Condition (Conditional Routing)

`tools_condition` from `langgraph.prebuilt` applies this logic:

```
if last message from LLM is a tool_call → route to "tools" node
if last message from LLM is NOT a tool_call → route to END
```

```python
from langgraph.prebuilt import tools_condition
```

> **Important:** The tools node must be named exactly `"tools"` for `tools_condition` to route to it correctly.

---

### 4.7 Full Code — Chatbot with Tools

```python
# ── Imports ────────────────────────────────────────────────────────────────
import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from IPython.display import Image, display

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

# ── 1. LLM ────────────────────────────────────────────────────────────────
llm = ChatGroq(model="llama3-8b-8192")

# ── 2. Tools ──────────────────────────────────────────────────────────────
tavily_tool = TavilySearch(max_results=2)

def multiply(a: int, b: int) -> int:
    """
    Multiply a and b.

    Args:
        a: First integer
        b: Second integer
    Returns:
        int: Product of a and b
    """
    return a * b

tools = [tavily_tool, multiply]

# ── 3. Bind tools to LLM ──────────────────────────────────────────────────
llm_with_tools = llm.bind_tools(tools)

# ── 4. State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── 5. Node definitions ───────────────────────────────────────────────────
def tool_calling_llm(state: State):
    """LLM node — decides whether to call a tool or answer directly."""
    return {"messages": llm_with_tools.invoke(state["messages"])}

# ── 6. Build Graph ────────────────────────────────────────────────────────
builder = StateGraph(State)

# Add nodes
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", tool_node)        # MUST be named "tools"

# Add edges
builder.add_edge(START, "tool_calling_llm")

# Conditional edge: tool call → tools node, no tool call → END
builder.add_conditional_edges("tool_calling_llm", tools_condition)

# After tool executes, go to END
builder.add_edge("tools", END)

# ── 7. Compile ────────────────────────────────────────────────────────────
graph = builder.compile()

# ── 8. Visualize ──────────────────────────────────────────────────────────
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass
# Output: START → tool_calling_llm → tools → END
#                               └──────────→ END
```

---

### 4.8 Invoking the Graph

```python
# Test 1 — triggers Tavily (web search)
response = graph.invoke({"messages": "What is the recent AI news?"})
for m in response["messages"]:
    m.pretty_print()
# HumanMessage: "What is the recent AI news?"
# AIMessage: (empty content, but tool_calls=[{"name": "tavily_search", ...}])
# ToolMessage: "Recent AI news: Nvidia self-driving... [results]"

# Test 2 — triggers multiply tool
response = graph.invoke({"messages": "What is 5 * 2?"})
for m in response["messages"]:
    m.pretty_print()
# AIMessage: tool_calls=[{"name": "multiply", "args": {"a": 5, "b": 2}}]
# ToolMessage: name=multiply, content=10

# Get last message content
print(response["messages"][-1].content)
```

---

## 5. Execution Flow Explained

### Simple Query (no tool needed)
```
Input: "What is Python?"
  → START
  → tool_calling_llm   (LLM answers directly, no tool_calls in response)
  → END                (tools_condition routes here when no tool call)
```

### Tool Query
```
Input: "What is recent AI news?"
  → START
  → tool_calling_llm   (LLM can't answer, sets tool_calls=[{tavily_search}])
  → tools              (tools_condition routes here, TavilySearch executes)
  → END
```

### Multi-query Problem (limitation of current graph)

```python
# This query has TWO questions:
response = graph.invoke({
    "messages": "Give me recent AI news AND multiply 5 by 10"
})
# Problem: After Tavily runs → goes to END
# The multiply question never gets answered!
# Solution: Route tools → back to LLM instead of END (→ ReAct Agent)
```

> This is the key limitation of the current graph. The fix is **ReAct architecture** — where tools route back to the LLM instead of END, letting the LLM handle multiple tool calls in sequence. This is covered next in the course.

---

## 6. Key Concepts Summary

### Graph API vs Functional API

| | Graph API | Functional API |
|---|---|---|
| Learning curve | Easier | Harder |
| Explicitness | Nodes, edges explicit | More abstracted |
| Recommendation | Start here | After mastering Graph API |

### `add_messages` Reducer Behavior

```python
# Without reducer (default) — replaces:
state["messages"] = new_message     # ❌ loses history

# With add_messages reducer — appends:
state["messages"].append(new_message)   # ✅ maintains history
```

### Node Naming — Critical Rule

```python
# tools_condition looks for a node named EXACTLY "tools"
builder.add_node("tools", tool_node)    # ✅ correct
builder.add_node("tool_node", tool_node)  # ❌ tools_condition won't find it
```

---

## 7. Quick Reference

```python
# Core imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Graph lifecycle
builder = StateGraph(State)
builder.add_node("node_name", node_function)
builder.add_edge(START, "node_name")
builder.add_edge("node_name", END)
builder.add_conditional_edges("node_name", condition_function)
graph = builder.compile()

# Running the graph
graph.invoke({"messages": "your input"})           # single call
graph.stream({"messages": "your input"})           # streaming

# Bind tools to LLM
llm_with_tools = llm.bind_tools([tool1, tool2])

# Tool node (executes tool calls made by LLM)
tool_node = ToolNode([tool1, tool2])

# Visualize
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

---

## 8. What's Coming Next (Course Roadmap)

**Part 1 (current):**
- ✅ Basic chatbot with StateGraph
- ✅ Chatbot with tools + conditional routing
- ⬜ ReAct agent (tools loop back to LLM)
- ⬜ Memory (short-term & checkpointing)
- ⬜ Human-in-the-loop in graphs
- ⬜ Streaming techniques
- ⬜ MCP (Model Context Protocol) from scratch

**Part 2 — Advanced LangGraph:**
- Multi-agent communication (agents talking to agents)
- Multi-state management
- Functional API
- LangGraph Studio + LangSmith debugging

**Part 3 — Production:**
- End-to-end projects
- LLM evaluation (MLflow, Grafana)
- Deployment (Hugging Face Spaces)