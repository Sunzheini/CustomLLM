#!/usr/bin/env python3
"""
Test client for the Hello World MCP Server
"""
import asyncio
import json
import subprocess
import sys
from typing import Optional


class SimpleMCPClient:
    """A simple MCP client to test our server."""

    def __init__(self, server_script: str):
        self.server_script = server_script
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0

    async def start(self):
        """Start the MCP server subprocess."""
        print(f"🚀 Starting server: {self.server_script}")
        print(f"   Using Python: {sys.executable}")
        self.process = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Start a thread to monitor stderr
        import threading
        def read_stderr():
            while True:
                line = self.process.stderr.readline()
                if not line:
                    break
                print(f"🔴 Server stderr: {line.strip()}")
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        # Give the server a moment to start
        await asyncio.sleep(0.5)

        # Initialize the connection
        await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })

        response = await self.read_response()
        print(f"✅ Server initialized: {response}")

        # Send initialized notification
        await self.send_notification("notifications/initialized")

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
        print(f"📤 Sent: {method}")
        return self.request_id

    async def send_notification(self, method: str):
        """Send a JSON-RPC notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
        message = json.dumps(notification) + "\n"
        self.process.stdin.write(message)
        self.process.stdin.flush()

    async def read_response(self) -> dict:
        """Read a response from the server."""
        # Check if process is still alive
        if self.process.poll() is not None:
            stderr_output = self.process.stderr.read()
            raise RuntimeError(f"Server process died! Exit code: {self.process.returncode}\nStderr: {stderr_output}")
        
        line = self.process.stdout.readline()
        if line:
            response = json.loads(line)
            print(f"📥 Received response (id={response.get('id')})")
            return response
        return {}

    async def list_tools(self):
        """Request the list of available tools."""
        await self.send_request("tools/list", {})
        return await self.read_response()

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the server."""
        await self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        return await self.read_response()

    def cleanup(self):
        """Stop the server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("🛑 Server stopped")


async def main():
    """Test the Hello World MCP server."""
    client = SimpleMCPClient("mcp_examples/hello_server.py")

    try:
        # Start and initialize
        await client.start()

        # List available tools
        print("\n" + "=" * 50)
        print("📋 Listing available tools...")
        print("=" * 50)
        tools_response = await client.list_tools()
        
        if 'result' in tools_response:
            tools = tools_response['result'].get('tools', [])
            for tool in tools:
                print(f"  🔧 Tool: {tool['name']}")
                print(f"     Description: {tool['description']}")

        # Call the say-hello tool
        print("\n" + "=" * 50)
        print("🔧 Calling 'say-hello' tool with name='Alice'...")
        print("=" * 50)
        result = await client.call_tool("say-hello", {"name": "Alice"})
        
        if 'result' in result:
            content = result['result'].get('content', [])
            for item in content:
                print(f"  💬 {item.get('text', '')}")

        print("\n" + "=" * 50)
        print("🎯 Testing with different name='Bob'...")
        print("=" * 50)
        result2 = await client.call_tool("say-hello", {"name": "Bob"})
        
        if 'result' in result2:
            content = result2['result'].get('content', [])
            for item in content:
                print(f"  💬 {item.get('text', '')}")
        
        print("\n" + "="*50)
        print("✨ All tests passed! Your MCP server works! ✨")
        print("="*50)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

# python mcp/test_hello_client.py