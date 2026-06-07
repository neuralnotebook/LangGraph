from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather") # Create a new MCP server with the name "Weather"


@mcp.tool()
async def get_weather(location: str) -> str:
    """get the current weather for a given location"""
    # In a real implementation, you would call a weather API here
    return f"The current weather in {location} is sunny with a high of 25°C."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")