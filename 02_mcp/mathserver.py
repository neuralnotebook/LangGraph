from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math") # Create a new MCP server with the name "Math"


@mcp.tool()
def add(a: int, b: int) -> int:
    """add two numbers together"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """multiply two numbers together"""
    return a * b

#The transport="stdio" argument tells the server to:
#Use standard input/output (stdin and stdout) to receive and respond to tool function calls.


if __name__ == "__main__":
    mcp.run(transport="stdio") # Run the MCP server using stdio transport