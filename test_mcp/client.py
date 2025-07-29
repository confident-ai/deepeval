import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from deepeval.test_case import ConversationalTestCase, MCPMetaData, Turn
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

mcp_data = []
turns = []

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        toolList = await self.session.list_tools()
        resourceList = await self.session.list_resources()
        promptList = await self.session.list_prompts()

        print("\nConnected to server with tools:", [tool.name for tool in toolList.tools])

        # Adding MCP meta data
        mcp_data.append(MCPMetaData(
            server_name=server_script_path,
            available_tools=toolList.tools,
            available_resources=resourceList.resources,
            available_prompts=promptList.prompts
        ))

    async def process_query(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        turns.append(Turn(role="user", content=query))

        response_text = []

        # Get list of available tools
        tool_response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in tool_response.tools]

        while True:
            # Claude API call
            response = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=messages,
                tools=available_tools
            )

            # Track tool uses and collect content
            tool_uses = []
            full_response_content = []

            for content in response.content:
                full_response_content.append(content)

                if content.type == 'text':
                    response_text.append(content.text)
                    turns.append(Turn(role="assistant", content=content.text))

                elif content.type == 'tool_use':
                    tool_uses.append(content)


            messages.append({
                "role": "assistant",
                "content": full_response_content
            })

            # 🔚 If no tool_use detected, we’re done
            if not tool_uses:
                break

            # 🛠 Handle tool calls and send results
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_args = tool_use.input
                tool_id = tool_use.id

                # Execute the tool
                result = await self.session.call_tool(tool_name, tool_args)
                prompt = await self.session.get_prompt("addingNote")
                resource = await self.session.read_resource("file:///project/src/notes.txt")

                # Track this tool usage in test logs
                turns.append(Turn(
                    role="assistant",
                    content=f"Tool call: {tool_name} with args {tool_args}",
                    mcp_prompts_called=[prompt],
                    mcp_tools_called=[result],
                    mcp_resources_called=[resource]
                ))

                # Send tool result message
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result.content
                    }]
                })

        return "\n".join(response_text)


    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            query = input("\nQuery: ").strip()

            if query.lower() == 'quit':
                convo_test_case = ConversationalTestCase(
                    turns=turns,
                    mcp_data=mcp_data
                )
                print(convo_test_case)
                print("-"*50)
                break

            response = await self.process_query(query)
            print("\n" + response)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())