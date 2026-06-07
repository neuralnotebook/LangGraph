
import os
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["02_mcp/mathserver.py"], # ensure correct absolute path 
                "transport": "stdio"
            },
            "weather": {
                "url": "http://localhost:8000/mcp",  # Ensure server is running here
                "transport": "streamable_http"
            }
        }
    )

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    tools = await client.get_tools()
    for tool in tools:
        print(tool.name)
        print(tool.description)
        print(tool.args_schema)
        print()
    # model = ChatGroq(model="llama-3.3-70b-versatile")
    model = ChatGroq(model="openai/gpt-oss-120b")
    agent = create_agent(model, tools)

    math_response = await agent.ainvoke(
        {"messages": [{
                "role": "system",
                "content": """
                Use tools one at a time.
                Never nest tool calls inside tool arguments.
                First call add, then use its result when calling multiply.
                """
            }, {"role": "user", "content": "what is (3 + 5) * 12"}]}
    )

    print("Math response:", math_response['messages'][-1].content) 

    weather_response = await agent.ainvoke(
        {"messages": [{
                "role": "system",
                "content": """
                Use tools one at a time.
                use weather to get the weather of the location.
                """
            }, {"role": "user", "content": "what is the weather in California?"}]}
    )
    print("Weather response:", weather_response['messages'][-1].content)


asyncio.run(main())