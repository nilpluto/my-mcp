import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


async def main():

    client = MultiServerMCPClient(
        {
            "calculator": {
                "command": "python",
                "args": ["./calculator_mcp.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    print("\nTOOLS FOUND:")
    for tool in tools:
        print("-", tool.name)

    llm = ChatOllama(
        model="qwen3:1.7b",
        temperature=0
    )

    agent = create_agent(
        llm,
        tools
    )

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Multiply 12 and 8. Return only the final number.",
                }
            ]
        }
    )

    print("\nALL MESSAGES:")
    for msg in response["messages"]:
        print("TYPE:", type(msg).__name__)
        print("CONTENT:", msg.content)
        print("TOOL CALLS:", getattr(msg, "tool_calls", None))
        print("-" * 50)

    print("\nFINAL ANSWER:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
