import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient, config

async def main():
    # Load environment variables
    load_dotenv()

    # Create MCPClient from config file
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    config_file="browser_mcp.json"


    print("Initializing MCPClient...")

    client = MCPClient.from_config_file(config_file)
    print("Initializing Chat...")
    llm = ChatGroq(model="llama-3.1-8b-instant")

    # Create agent with the client
    agent = MCPAgent(llm=llm, client=client, max_steps=15)

    print("\nInteracting with the agent...")
    print("Type exit to end the conversation")
    print("Type clear to clear the conversation")
    print("--------------------------------\n")

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting...")
                break
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared...")
                continue

            print("Assistant: ", end="",flush=True)
            
            try:
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"Error: {e}")
    finally:
        if client and client.sessions:
            await client.close_all_sessions()

        print("\nThank you for using the agent! Goodbye!")
        print("--------------------------------\n")

if __name__ == "__main__":
    asyncio.run(main())