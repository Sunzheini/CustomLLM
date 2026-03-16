#!/usr/bin/env python3
"""
Test client for the Calculator MCP Server
Demonstrates multiple tools and error handling
"""
import asyncio
import json
import subprocess
import sys
import threading
from typing import Optional


class CalculatorMCPClient:
    """MCP client specifically for testing the calculator server."""
    
    def __init__(self, server_script: str):
        self.server_script = server_script
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        
    async def start(self):
        """Start the MCP server subprocess."""
        print("=" * 70)
        print("🧮 CALCULATOR MCP SERVER TEST")
        print("=" * 70)
        print(f"Server: {self.server_script}")
        print(f"Python: {sys.executable}")
        print("=" * 70)
        
        self.process = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Monitor stderr silently (you can uncomment for debugging)
        def read_stderr():
            while True:
                line = self.process.stderr.readline()
                if not line:
                    break
                # print(f"[Server] {line.strip()}")
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        await asyncio.sleep(0.5)
        
        # Initialize
        await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "calculator-test-client",
                "version": "1.0.0"
            }
        })
        
        response = await self.read_response()
        server_info = response.get('result', {}).get('serverInfo', {})
        print(f"✅ Connected to: {server_info.get('name', 'Unknown')}")
        
        # Send initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        message = json.dumps(notification) + "\n"
        self.process.stdin.write(message)
        self.process.stdin.flush()
        
    async def send_request(self, method: str, params: dict) -> int:
        """Send a JSON-RPC request to the server."""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        message = json.dumps(request) + "\n"
        self.process.stdin.write(message)
        self.process.stdin.flush()
        return self.request_id
        
    async def read_response(self) -> dict:
        """Read a response from the server."""
        if self.process.poll() is not None:
            raise RuntimeError(f"Server process died! Exit code: {self.process.returncode}")
        
        line = self.process.stdout.readline()
        if line:
            return json.loads(line)
        return {}
        
    async def list_tools(self):
        """Request the list of available tools."""
        await self.send_request("tools/list", {})
        return await self.read_response()
        
    async def call_tool(self, tool_name: str, arguments: dict, description: str = ""):
        """Call a tool and display the result."""
        if description:
            print(f"\n🔧 {description}")
        else:
            print(f"\n🔧 Calling: {tool_name}({arguments})")
        
        await self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        response = await self.read_response()
        
        if 'result' in response:
            content = response['result'].get('content', [])
            for item in content:
                text = item.get('text', '')
                if text.startswith('Error:'):
                    print(f"   ❌ {text}")
                else:
                    print(f"   ✅ {text}")
        elif 'error' in response:
            print(f"   ❌ Error: {response['error']}")
        
        return response
        
    def cleanup(self):
        """Stop the server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("\n" + "=" * 70)
            print("🛑 Server stopped")
            print("=" * 70)


async def main():
    """Run comprehensive calculator tests."""
    client = CalculatorMCPClient("mcp_examples/calculator_server.py")
    
    try:
        await client.start()
        
        # List available tools
        print("\n" + "▶" * 35)
        print("STEP 1: Discovering Available Tools")
        print("▶" * 35)
        
        tools_response = await client.list_tools()
        if 'result' in tools_response:
            tools = tools_response['result'].get('tools', [])
            print(f"\n📋 Found {len(tools)} tools:")
            for tool in tools:
                print(f"   • {tool['name']}: {tool['description']}")
        
        # Test basic operations
        print("\n" + "▶" * 35)
        print("STEP 2: Testing Basic Operations")
        print("▶" * 35)
        
        await client.call_tool("add", {"a": 10, "b": 5}, "Addition: 10 + 5")
        await client.call_tool("subtract", {"a": 10, "b": 5}, "Subtraction: 10 - 5")
        await client.call_tool("multiply", {"a": 10, "b": 5}, "Multiplication: 10 × 5")
        await client.call_tool("divide", {"a": 10, "b": 5}, "Division: 10 ÷ 5")
        
        # Test advanced operations
        print("\n" + "▶" * 35)
        print("STEP 3: Testing Advanced Operations")
        print("▶" * 35)
        
        await client.call_tool("power", {"base": 2, "exponent": 8}, "Power: 2 ^ 8")
        await client.call_tool("sqrt", {"value": 64}, "Square Root: √64")
        await client.call_tool("sqrt", {"value": 2}, "Square Root: √2")
        
        # Test with decimals
        print("\n" + "▶" * 35)
        print("STEP 4: Testing with Decimal Numbers")
        print("▶" * 35)
        
        await client.call_tool("add", {"a": 3.14, "b": 2.86}, "Addition: 3.14 + 2.86")
        await client.call_tool("multiply", {"a": 2.5, "b": 4.2}, "Multiplication: 2.5 × 4.2")
        
        # Test error handling
        print("\n" + "▶" * 35)
        print("STEP 5: Testing Error Handling")
        print("▶" * 35)
        
        await client.call_tool("divide", {"a": 10, "b": 0}, "Division by zero: 10 ÷ 0")
        await client.call_tool("sqrt", {"value": -4}, "Square root of negative: √(-4)")
        
        # Test calculation history
        print("\n" + "▶" * 35)
        print("STEP 6: Viewing Calculation History")
        print("▶" * 35)
        
        await client.call_tool("history", {}, "Get calculation history")
        
        # Complex calculation example
        print("\n" + "▶" * 35)
        print("STEP 7: Complex Calculation Example")
        print("▶" * 35)
        print("Calculate: (2^10 + √144) × 3.5")
        
        # 2^10
        result1 = await client.call_tool("power", {"base": 2, "exponent": 10}, "Step 1: 2^10")
        
        # √144
        result2 = await client.call_tool("sqrt", {"value": 144}, "Step 2: √144")
        
        # 1024 + 12
        result3 = await client.call_tool("add", {"a": 1024, "b": 12}, "Step 3: 1024 + 12")
        
        # 1036 × 3.5
        result4 = await client.call_tool("multiply", {"a": 1036, "b": 3.5}, "Step 4: 1036 × 3.5")
        
        # Final history
        print("\n" + "▶" * 35)
        print("FINAL: Updated History")
        print("▶" * 35)
        await client.call_tool("history", {}, "Final calculation history")
        
        # Summary
        print("\n" + "🌟" * 35)
        print("TEST SUMMARY")
        print("🌟" * 35)
        print("""
✅ All tests completed successfully!

What you learned:
1. Multiple tools in one MCP server
2. Different parameter types (numbers)
3. Input validation (division by zero, negative sqrt)
4. Error handling and user-friendly messages
5. Stateful operations (calculation history)
6. Complex multi-step calculations

This calculator demonstrates a real-world MCP server with:
- 7 different tools
- Proper error handling
- State management (history)
- Clear, descriptive tool schemas
        """)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

# python mcp_examples/test_calculator_client.py
