#!/usr/bin/env python3
"""
Calculator MCP Server
A practical MCP server with multiple mathematical operation tools.

This demonstrates:
- Multiple tools in one server
- Input validation
- Error handling
- Different parameter types (numbers)
- Documentation
"""
import asyncio
import logging
import math
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the MCP server instance
app = Server("calculator-server")

# Store calculation history
calculation_history = []


def add_to_history(operation: str, result: float):
    """Add a calculation to history."""
    calculation_history.append({
        "operation": operation,
        "result": result
    })
    # Keep only last 10
    if len(calculation_history) > 10:
        calculation_history.pop(0)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available calculator tools."""
    logger.info("Client requested calculator tools")
    
    return [
        Tool(
            name="add",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="subtract",
            description="Subtract second number from first number (a - b)",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "Number to subtract from"
                    },
                    "b": {
                        "type": "number",
                        "description": "Number to subtract"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="multiply",
            description="Multiply two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="divide",
            description="Divide first number by second number (a / b)",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "Numerator (dividend)"
                    },
                    "b": {
                        "type": "number",
                        "description": "Denominator (divisor) - cannot be zero"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="power",
            description="Raise a number to a power (a ^ b)",
            inputSchema={
                "type": "object",
                "properties": {
                    "base": {
                        "type": "number",
                        "description": "Base number"
                    },
                    "exponent": {
                        "type": "number",
                        "description": "Exponent (power)"
                    }
                },
                "required": ["base", "exponent"]
            }
        ),
        Tool(
            name="sqrt",
            description="Calculate square root of a number",
            inputSchema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Number to get square root of (must be non-negative)"
                    }
                },
                "required": ["value"]
            }
        ),
        Tool(
            name="history",
            description="Get the last 10 calculations performed",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool execution with proper error handling."""
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    
    try:
        if name == "add":
            a = float(arguments["a"])
            b = float(arguments["b"])
            result = a + b
            add_to_history(f"{a} + {b}", result)
            return [TextContent(
                type="text",
                text=f"Result: {a} + {b} = {result}"
            )]
            
        elif name == "subtract":
            a = float(arguments["a"])
            b = float(arguments["b"])
            result = a - b
            add_to_history(f"{a} - {b}", result)
            return [TextContent(
                type="text",
                text=f"Result: {a} - {b} = {result}"
            )]
            
        elif name == "multiply":
            a = float(arguments["a"])
            b = float(arguments["b"])
            result = a * b
            add_to_history(f"{a} × {b}", result)
            return [TextContent(
                type="text",
                text=f"Result: {a} × {b} = {result}"
            )]
            
        elif name == "divide":
            a = float(arguments["a"])
            b = float(arguments["b"])
            
            if b == 0:
                return [TextContent(
                    type="text",
                    text="Error: Division by zero is not allowed!"
                )]
            
            result = a / b
            add_to_history(f"{a} ÷ {b}", result)
            return [TextContent(
                type="text",
                text=f"Result: {a} ÷ {b} = {result}"
            )]
            
        elif name == "power":
            base = float(arguments["base"])
            exponent = float(arguments["exponent"])
            result = math.pow(base, exponent)
            add_to_history(f"{base} ^ {exponent}", result)
            return [TextContent(
                type="text",
                text=f"Result: {base} ^ {exponent} = {result}"
            )]
            
        elif name == "sqrt":
            value = float(arguments["value"])
            
            if value < 0:
                return [TextContent(
                    type="text",
                    text="Error: Cannot calculate square root of negative number!"
                )]
            
            result = math.sqrt(value)
            add_to_history(f"√{value}", result)
            return [TextContent(
                type="text",
                text=f"Result: √{value} = {result}"
            )]
            
        elif name == "history":
            if not calculation_history:
                return [TextContent(
                    type="text",
                    text="No calculation history available."
                )]
            
            history_text = "Recent calculations:\n"
            for i, calc in enumerate(calculation_history, 1):
                history_text += f"{i}. {calc['operation']} = {calc['result']}\n"
            
            return [TextContent(
                type="text",
                text=history_text.strip()
            )]
            
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except KeyError as e:
        logger.error(f"Missing required parameter: {e}")
        return [TextContent(
            type="text",
            text=f"Error: Missing required parameter {e}"
        )]
    except ValueError as e:
        logger.error(f"Invalid parameter value: {e}")
        return [TextContent(
            type="text",
            text=f"Error: Invalid parameter value - {e}"
        )]
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def main():
    """Run the Calculator MCP server."""
    logger.info("Starting Calculator MCP Server...")
    logger.info("Available operations: add, subtract, multiply, divide, power, sqrt, history")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

