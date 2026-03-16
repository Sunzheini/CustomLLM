#!/usr/bin/env python3
"""
A simple Hello World MCP Server
This is the most basic MCP server possible.
"""
import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Create the MCP server instance
app = Server("hello-world-server")


# Define what tools are available
@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    logger.info("Client requested tool list")
    return [
        Tool(
            name="say-hello",
            description="Says hello to a person by name",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the person to greet"
                    }
                },
                "required": ["name"]
            }
        )
    ]


# Handle tool execution
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from the client."""
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    if name == "say-hello":
        person_name = arguments.get("name", "World")
        greeting = f"Hello, {person_name}! Welcome to your first MCP server! 🎉"
        return [TextContent(type="text", text=greeting)]
    else:
        raise ValueError(f"Unknown tool: {name}")


# Main entry point
async def main():
    """Run the MCP server using stdio transport."""
    logger.info("Starting Hello World MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
