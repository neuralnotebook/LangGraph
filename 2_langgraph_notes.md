# LangGraph — Part 2: ReAct Agent, Memory, Streaming & Human-in-the-Loop

> Continuation of Krishna's LangGraph crash course. Covers the ReAct agent architecture, persistent memory with checkpointing, streaming modes, and human-in-the-loop interrupts.

---

## 1. The Problem with `tools → END` Graph

In the previous graph (tools → END), when a query contains **multiple questions**, only the first tool call gets answered.

```python
# This query has TWO tasks:
graph.invoke({"messages": "Give me recent AI news AND multiply 5 by 10"})

# What happens:
# → LLM calls TavilySearch (for AI news) ✅
# → tools → END (graph terminates)
# → multiply 5*10 is NEVER answered ❌
```

**Root cause:** After the tool executes, the result goes straight to `END` instead of back to the LLM.

**Fix:** Route `tools → tool_calling_llm` instead of `tools → END` → this is the **ReAct architecture**.

---

## 2. ReAct Agent Architecture

**ReAct = Reason + Act + Observe**

| Phase | What Happens |
|---|---|
| **Act** | LLM receives input → makes a tool call |
| **Observe** | Tool executes → result returned to LLM |
| **Reason** | LLM decides: "Is there more to do? Make another tool call OR go to END" |

This loop continues until the LLM decides there's nothing left to do.

### Graph Structure — ReAct

```
START → [tool_calling_llm] ──(tool call?)──→ [tools] ──→ [tool_calling_llm]
                            └──(no tool call)──────────────────────────→ END
```

The key change: `tools → tool_calling_llm` instead of `tools → END`.

---

### Full ReAct Agent Code

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
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# ── 3. State ──────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── 4. Node definition ────────────────────────────────────────────────────
def tool_calling_llm(state: State):
    return {"messages": llm_with_tools.invoke(state["messages"])}

# ── 5. Build Graph ────────────────────────────────────────────────────────
builder = StateGraph(State)

builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", tool_node)

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)

# ⭐ KEY CHANGE: tools → tool_calling_llm (not END)
builder.add_edge("tools", "tool_calling_llm")

graph = builder.compile()

# ── 6. Visualize ──────────────────────────────────────────────────────────
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass
```

### Test the ReAct Agent

```python
# Multi-task query — now BOTH get answered
response = graph.invoke({
    "messages": "Give me recent AI news AND multiply 5 by 10"
})

for m in response["messages"]:
    m.pretty_print()

# Flow:
# HumanMessage: "Give me recent AI news AND multiply 5 by 10"
# AIMessage: tool_calls=[{tavily_search: "recent AI news"}]
# ToolMessage: "...AI news results..."
# AIMessage: tool_calls=[{multiply: {a:5, b:10}}]    ← LLM picks up second task
# ToolMessage: name=multiply, content=50
# AIMessage: "Here's the recent AI news: [...] And 5 × 10 = 50."
```

> The LLM keeps looping through tools until ALL tasks in the input are resolved, then generates a final combined response and routes to `END`.

---

## 3. Memory — Persistent Checkpointing

### The Problem Without Memory

```python
# Turn 1
graph.invoke({"messages": "Hi, my name is Kush"})
# → "Nice to meet you Kush!"

# Turn 2 (new invoke call)
graph.invoke({"messages": "What is my name?"})
# → "I don't have any information about your name." ❌
```

Each `graph.invoke()` call is **stateless** — the graph has no memory of previous interactions.

### The Fix — MemorySaver (In-Memory Checkpoint)

`MemorySaver` saves the entire state (all messages) after every node execution. When you pass the same `thread_id`, it loads the previous state and continues from where it left off.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

Pass it to `compile()` via the `checkpointer` parameter:

```python
graph = builder.compile(checkpointer=memory)
```

### Thread ID — Unique Session per User

A `thread_id` uniquely identifies a conversation session. Same `thread_id` = same conversation history loaded.

```python
# Create a config with a unique thread ID
config = {"configurable": {"thread_id": "user_session_001"}}
```

### Full Memory Example

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# ⭐ Pass memory as checkpointer
graph = builder.compile(checkpointer=memory)

# Session config — unique per user
config = {"configurable": {"thread_id": "user_001"}}

# Turn 1
response = graph.invoke(
    {"messages": "Hi, my name is Kush"},
    config=config                          # ⭐ always pass config
)
print(response["messages"][-1].content)
# → "Nice to meet you Kush! How can I help you today?"

# Turn 2 — same config → loads previous conversation
response = graph.invoke(
    {"messages": "What is my name?"},
    config=config
)
print(response["messages"][-1].content)
# → "Your name is Kush!" ✅

# Turn 3 — still remembers
response = graph.invoke(
    {"messages": "Do you remember my name?"},
    config=config
)
print(response["messages"][-1].content)
# → "Yes, your name is Kush!" ✅
```

> **Key rule:** Always pass the same `config` to maintain the conversation thread. Different `thread_id` = fresh conversation.

### MemorySaver Internals

```python
# What MemorySaver does internally (simplified):
# After each node execution → saves state to a dict keyed by thread_id
# {
#   "user_001": {
#       "messages": [
#           HumanMessage("Hi, my name is Kush"),
#           AIMessage("Nice to meet you Kush!"),
#           HumanMessage("What is my name?"),
#           AIMessage("Your name is Kush!"),
#       ]
#   }
# }
```

---

## 4. Streaming in LangGraph

LangGraph provides two ways to run a graph and two streaming modes.

### Methods: `stream` vs `astream`

| Method | Type | Use Case |
|---|---|---|
| `graph.stream()` | Synchronous | Regular Python code |
| `graph.astream()` | Asynchronous | Async Python code (`async/await`) |

### Modes: `updates` vs `values`

Given a graph with nodes: `node1 → node2 → node3`

**`mode="updates"` — only the latest node's output**

```
node1 executes → prints: {"messages": [AIMessage("Hi")]}
node2 executes → prints: {"messages": [AIMessage("my name is")]}
node3 executes → prints: {"messages": [AIMessage("Kush")]}
```
Only the **delta** (what changed in that node) is shown each time.

**`mode="values"` — cumulative full state after each node**

```
node1 executes → prints: [HumanMessage("..."), AIMessage("Hi")]
node2 executes → prints: [HumanMessage("..."), AIMessage("Hi"), AIMessage("my name is")]
node3 executes → prints: [HumanMessage("..."), AIMessage("Hi"), AIMessage("my name is"), AIMessage("Kush")]
```
The **full appended message list** is shown after each node.

---

### Code Examples

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "stream_demo_1"}}
input_msg = {"messages": "Hi, my name is Kush. I like cricket."}
```

**Mode: `updates`** — see only what each node produced

```python
for chunk in graph.stream(input_msg, config=config, stream_mode="updates"):
    print(chunk)

# Output:
# {"superbot": {"messages": [AIMessage("Nice to meet you Kush!...")]}}
# Only the AI response — no human message shown
```

**Mode: `values`** — see full state after every node

```python
for chunk in graph.stream(input_msg, config=config, stream_mode="values"):
    print(chunk)

# Output:
# {"messages": [HumanMessage("Hi, my name is Kush...")]}         ← after input
# {"messages": [HumanMessage("..."), AIMessage("Nice to meet you Kush!")]}  ← after chatbot node
# Full conversation history shown at each step
```

**`astream_events`** — detailed debugging, every internal event

```python
config = {"configurable": {"thread_id": "stream_demo_5"}}

async for event in graph.astream_events(
    {"messages": "Hi, my name is Kush. I like cricket."},
    config=config,
    version="v1"
):
    print(event)
# Much more verbose — shows every internal LangGraph event
# Useful for debugging complex multi-node graphs
```

### When to Use Which

| Scenario | Use |
|---|---|
| Building a chatbot UI | `stream` with `mode="updates"` |
| Debugging graph execution | `stream` with `mode="values"` |
| Deep event-level debugging | `astream_events` |
| Async web framework (FastAPI) | `astream` |

---

## 5. Human-in-the-Loop (Graph-Level)

> Note: This is different from LangChain middleware's Human-in-the-Loop. Here, the interrupt is implemented **as a tool** inside the graph.

### Concept

```
START → [chatbot] ──(human_assistance tool call?)──→ [tools] ──interrupt──→ HUMAN INPUT
                                                                         ↓
                                                     resume with Command → [tools] → [chatbot] → END
```

The `interrupt()` function **pauses the graph mid-execution** and waits for a human to provide input before resuming.

### Imports

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from typing import Annotated
from typing_extensions import TypedDict
```

### Define the Human Assistance Tool

```python
@tool
def human_assistance(query: str) -> str:
    """
    Request assistance from a human.
    Use this when you need expert human input or approval.

    Args:
        query: The question or request to send to the human
    Returns:
        str: The human's response
    """
    # interrupt() pauses the entire graph and surfaces the query to the user
    human_response = interrupt({"query": query})
    return human_response["data"]
```

> `interrupt(value)` — freezes graph execution at this point. The `value` is what gets surfaced to the calling code so a human can see it.

### Build the Graph

```python
# Tools setup
tavily_tool = TavilySearch(max_results=2)
tools = [tavily_tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Node
def chatbot(state: State):
    return {"messages": llm_with_tools.invoke(state["messages"])}

# Graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")   # ReAct loop

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

---

### Execution Flow

#### Step 1 — Trigger the graph (will pause at interrupt)

```python
config = {"configurable": {"thread_id": "hitl_session_1"}}

user_input = "I need some expert guidance for building AI agents. Could you request assistance for me?"

# Stream until interrupt fires
for chunk in graph.stream(
    {"messages": user_input},
    config=config,
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()

# Output:
# HumanMessage: "I need some expert guidance..."
# AIMessage: tool_calls=[{human_assistance: {"query": "expert guidance for AI agents"}}]
# ⏸ GRAPH PAUSES HERE — waiting for human input
```

#### Step 2 — Human provides response and resumes

```python
# Human types their response
human_response = "We the experts are here to help you out. We recommend you check out LangGraph to build your agents. It's much more reliable and extensible than simple autonomous agents."

# Resume the graph with the human's response
for chunk in graph.stream(
    Command(resume={"data": human_response}),   # ⭐ Command resumes the graph
    config=config,                              # ⭐ same config = same thread
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()

# Output:
# ToolMessage: "We the experts are here to help... check out LangGraph..."
# AIMessage: "Thank you for the recommendation! LangGraph seems like a great tool..."
```

#### Step 3 — Continue the loop (optional)

```python
# Can interrupt and resume as many times as needed
# LLM may ask another follow-up → triggers another interrupt → human responds again
```

---

### `interrupt` vs `Command` — Summary

| | `interrupt(value)` | `Command(resume=value)` |
|---|---|---|
| **Where used** | Inside the tool function | In the calling code to resume |
| **Effect** | Freezes graph execution | Resumes frozen graph |
| **What's passed** | Dict shown to human (the question) | Dict returned to the tool (the answer) |
| **Import from** | `langgraph.types` | `langgraph.types` |

---

## 6. Graph Patterns — Summary

| Pattern | Edge from tools | Use Case |
|---|---|---|
| **Simple tool graph** | `tools → END` | Single tool call, one question |
| **ReAct agent** | `tools → tool_calling_llm` | Multi-tool, multi-question inputs |
| **With memory** | + `checkpointer=MemorySaver()` | Persistent multi-turn conversations |
| **Human-in-the-Loop** | + `interrupt()` tool + `Command(resume=...)` | Human approval/input mid-execution |

---

## 7. Quick Reference

```python
# ── ReAct: route tools back to LLM ────────────────────────────────────────
builder.add_edge("tools", "tool_calling_llm")    # not END!

# ── Memory ────────────────────────────────────────────────────────────────
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "unique_user_id"}}
graph.invoke({"messages": "..."}, config=config)

# ── Streaming modes ───────────────────────────────────────────────────────
# updates = only what changed per node
graph.stream(input, config, stream_mode="updates")
# values = full state after every node
graph.stream(input, config, stream_mode="values")
# detailed event debugging
graph.astream_events(input, config, version="v1")

# ── Human-in-the-Loop ─────────────────────────────────────────────────────
from langgraph.types import interrupt, Command

# In tool: pause and ask human
human_response = interrupt({"query": "your question here"})

# In calling code: resume with human's answer
graph.stream(Command(resume={"data": "human answer"}), config=config)
```