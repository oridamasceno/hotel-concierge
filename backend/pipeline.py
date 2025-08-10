import asyncio
import os
from dotenv import load_dotenv

from mcp_agents import hotel_concierge_agent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def process_hotel_query(user_message: str):
    """
    Processes a user query using the hotel concierge agent.
    """
    if not OPENAI_API_KEY:
        return "Error: OPENAI_API_KEY not found in environment variables."

    response = await hotel_concierge_agent(user_message)
    return response

if __name__ == "__main__":
    async def main():
        print("Hotel Concierge Chatbot Backend")
        print("Type your queries. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = await process_hotel_query(user_input)
            print(f"Bot: {response}")

    asyncio.run(main())
