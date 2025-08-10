import os
import asyncio
import sys
import traceback
import platform
from textwrap import dedent
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.tools.mcp import MCPTools, MultiMCPTools
import json
from agno.utils.pprint import apprint_run_response

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def load_hotel_data(file_path="hotel_data.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return None

hotel_data = load_hotel_data()


# windows compatibility fix
if platform.system() == "Windows":
    # setting the event loop policy for Windows
    if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    elif hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def extract_text_from_response(response_stream):
    """
    Extract text content from the response stream object
    """
    try:
        if hasattr(response_stream, 'content'):
            return response_stream.content
        elif hasattr(response_stream, 'messages') and response_stream.messages:
            last_message = response_stream.messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                return last_message['content']
        elif hasattr(response_stream, 'text'):
            return response_stream.text
        elif isinstance(response_stream, str):
            return response_stream
        else:
            return str(response_stream)
    except Exception as e:
        print(f"Error extracting text from response: {e}")
        return None

def get_mcp_command():
    """
    Get the appropriate MCP command based on the platform
    """
    if platform.system() == "Windows":
        # trying different approaches for Windows - credits Claude 4
        commands_to_try = [
            "npx -y @modelcontextprotocol/server-google-maps",
            "npx.cmd -y @modelcontextprotocol/server-google-maps",
            "node_modules/.bin/npx -y @modelcontextprotocol/server-google-maps"
        ]
    else:
        commands_to_try = ["npx -y @modelcontextprotocol/server-google-maps"]
    
    return commands_to_try

async def test_mcp_connection():
    """
    Test if MCP tools can be initialized properly
    """
    commands = get_mcp_command()
    for cmd in commands:
        try:
            print(f"Testing MCP command: {cmd}")
            mcp_tools = MCPTools(cmd)
            async with mcp_tools:
                print(f"Successfully connected with command: {cmd}")
                return cmd
        except Exception as e:
            print(f"Failed with command {cmd}: {e}")
            continue
    return None


## transport agent
async def transport_mcp_agent(message: str, people: int = 1):
    response_text = None
    agent = None
    
    try:
        # testing MCP connection
        working_cmd = await test_mcp_connection()
        if not working_cmd:
            print("No working MCP command found, using fallback agent without MCP tools")
            return await transport_fallback_agent(message, people)
        
        print(f"Using MCP command: {working_cmd}")
        mcp_tools = MCPTools(working_cmd)
        
        async with mcp_tools:
            print("MCPTools context entered successfully")
            
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
                instructions=dedent(f"""\
                    You are a travel agent. Your task is to find the best transport options for {people} number of people.\
                    You will find the exact time and cost of traveling (convert to USD) - for each of the following modes:\
                    You must provide the exact correct time and cost of these routes. You must not return any incorrect responses.\
                    - Car\
                    - Train (you must mention railway station name)\
                    - Flight (you must mention airport name)\
                    You will use the Google Maps API to find the best transport options.\
                    You must scan all the nearby airports, railway stations and must return a valid transport option.\
                    You will return the response in JSON format.\
                    You will not return any unnecessary text.
                    """
                ),
                tools=[mcp_tools],
                markdown=True,
            )
            print("Transport Agent initialized!")

            try:
                response_stream = await agent.arun(message, stream=False)
                await apprint_run_response(response_stream, markdown=True)
                response_text = extract_text_from_response(response_stream)
                
            finally:
                if hasattr(agent, 'close') and callable(agent.close):
                    try:
                        if asyncio.iscoroutinefunction(agent.close):
                            await agent.close()
                        else:
                            agent.close()
                        print("Agent closed successfully")
                    except Exception as e:
                        print(f"Error closing agent: {e}")
        
        print("MCPTools context exited.")

    except Exception as e:
        # fallback to agent without MCP tools
        print("Falling back to agent without MCP tools")
        response_text = await transport_fallback_agent(message, people)
    
    return response_text

## fallback transport agent
async def transport_fallback_agent(message: str, people: int = 1):
    """
    Fallback transport agent without MCP tools
    """
    try:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        agent = Agent(
            model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
            instructions=dedent(f"""\
                You are a travel agent. Your task is to find the best transport options for {people} number of people.\
                You will find the exact time and cost of traveling (convert to USD) - for each of the following modes:\
                You must provide the exact correct time and cost of these routes. You must not return any incorrect responses.\
                - Car\
                - Train (you must mention railway station name)\
                - Flight (you must mention airport name)\
                You will use the Google Maps API to find the best transport options.\
                You must scan all the nearby airports, railway stations and must return a valid transport option.\
                You will return the response in JSON format.\
                You will not return any unnecessary text.
                """
            ),
            markdown=True,
        )
        
        response_stream = await agent.arun(message, stream=False)
        response_text = extract_text_from_response(response_stream)
        
        
        if hasattr(agent, 'close') and callable(agent.close):
            try:
                if asyncio.iscoroutinefunction(agent.close):
                    await agent.close()
                else:
                    agent.close()
            except Exception as e:
                print(f"Error closing fallback agent: {e}")
        
        return response_text
        
    except Exception as e:
        print(f"Error in fallback transport agent: {e}")
        return None


def get_checkin_checkout_times(hotel_data):
    if hotel_data and "faqs" in hotel_data:
        for faq in hotel_data["faqs"]:
            if "check-in" in faq["question"].lower() and "check-out" in faq["question"].lower():
                return faq["answer"]
    return "I'm sorry, I don't have information about check-in and check-out times at the moment."

def get_amenities(hotel_data):
    if hotel_data and "amenities" in hotel_data:
        return "Our hotel offers the following amenities: " + ", ".join(hotel_data["amenities"]) + "."
    return "I'm sorry, I don't have information about amenities at the moment."

def get_room_types(hotel_data):
    if hotel_data and "room_types" in hotel_data:
        response = "We have several room types available:\n"
        for room in hotel_data["room_types"]:
            response += f"- {room['name']}: {room['description']} (Price range: {room['price_range']})\n"
        return response
    return "I'm sorry, I don't have information about room types at the moment."

def get_faq_answer(question, hotel_data):
    if hotel_data and "faqs" in hotel_data:
        for faq in hotel_data["faqs"]:
            if question.lower() in faq["question"].lower():
                return faq["answer"]
    return "I'm sorry, I couldn't find an answer to that question in our FAQs."

def get_contact_info(hotel_data):
    if hotel_data and "contact_info" in hotel_data:
        phone = hotel_data["contact_info"].get("phone", "N/A")
        email = hotel_data["contact_info"].get("email", "N/A")
        return f"You can contact us at phone: {phone} or email: {email}."
    return "I'm sorry, I don't have contact information at the moment."

def get_booking_link(hotel_data):
    if hotel_data and "booking_link" in hotel_data:
        return f"You can book your stay through our official booking link: {hotel_data['booking_link']}"
    return "I'm sorry, I don't have a booking link at the moment."

## hotel concierge agent
async def hotel_concierge_agent(message: str):
    response_text = None
    try:
        if not hotel_data:
            return "I'm sorry, I don't have hotel information loaded at the moment."

        # Basic intent recognition
        message_lower = message.lower()
        if "check-in" in message_lower or "check-out" in message_lower:
            response_text = get_checkin_checkout_times(hotel_data)
        elif "amenities" in message_lower or "facilities" in message_lower:
            response_text = get_amenities(hotel_data)
        elif "room types" in message_lower or "rooms available" in message_lower:
            response_text = get_room_types(hotel_data)
        elif "contact" in message_lower or "phone" in message_lower or "email" in message_lower:
            response_text = get_contact_info(hotel_data)
        elif "book" in message_lower or "reservation" in message_lower:
            response_text = get_booking_link(hotel_data)
        else:
            # Try to find an answer in FAQs
            faq_answer = get_faq_answer(message, hotel_data)
            if faq_answer != "I'm sorry, I couldn't find an answer to that question in our FAQs.":
                response_text = faq_answer
            else:
                # Fallback to a general LLM if no specific intent is matched
                agent = Agent(
                    model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
                    instructions=dedent(f"""\
                        You are a helpful hotel concierge chatbot. Provide concise and accurate information based on the user's query.
                        If you don't have specific information, politely state that you cannot assist with that particular request.
                        Do not make up information.
                        """
                    ),
                    markdown=True,
                )
                response_stream = await agent.arun(message, stream=False)
                response_text = extract_text_from_response(response_stream)
                if hasattr(agent, 'close') and callable(agent.close):
                    try:
                        if asyncio.iscoroutinefunction(agent.close):
                            await agent.close()
                        else:
                            agent.close()
                    except Exception as e:
                        print(f"Error closing agent: {e}")
        
    except Exception as e:
        print(f"Error occurred in hotel_concierge_agent: {e}")
        response_text = "I'm sorry, an error occurred while processing your request."
        
    return response_text



## sightseeing agent
async def sightseeing_mcp_agent(message: str, place: str, people: int = 1):
    response_text = None
    try:
        working_cmd = await test_mcp_connection()
        if not working_cmd:
            print("No working MCP command found, using fallback agent without MCP tools")
            return await sightseeing_fallback_agent(message, place, people)
        
        async with MCPTools(working_cmd) as mcptools:
            
            print("MCPTools initializing for Sightseeing Agent...")
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
                instructions=dedent(f"""\
                You are a local tour guide for {place}. Your task is to recommend the best sightseeing locations.\
                There are {people} people in the group.\
                
                Provide specific recommendations including:\
                - Top 4-5 must-visit attractions\
                - Brief description of each place\
                - Approximate visit duration\
                - Entry fees (if any, in USD)\
                - Best time to visit\
                
                Focus on attractions within and close to {place}.\
                Return the response in clear text format.
                """
                ),
                tools=[mcptools],
                markdown=True,
            )
            print("Sightseeing Agent initialized!")
            
            try:
                response_stream = await agent.arun(message, stream=False)
                await apprint_run_response(response_stream, markdown=True)
                
                # Extract the actual text content
                response_text = extract_text_from_response(response_stream)
                
            finally:
                if hasattr(agent, 'close') and callable(agent.close):
                    try:
                        if asyncio.iscoroutinefunction(agent.close):
                            await agent.close()
                        else:
                            agent.close()
                        print("Agent closed")
                    except Exception as e:
                        print(f"Error closing agent: {e}")
                    
        print("MCPTools context exited.")
    
    except Exception as e:
        print(f"Error occurred in sightseeing_agent: {e}")
        response_text = await sightseeing_fallback_agent(message, place, people)
        
    return response_text

## fallback sightseeing agent
async def sightseeing_fallback_agent(message: str, place: str, people: int = 1):
    """
    Fallback sightseeing agent without MCP tools
    """
    try:
        agent = Agent(
            model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
            instructions=dedent(f"""\
                You are a local tour guide for {place}. Your task is to recommend the best sightseeing locations.\
                There are {people} people in the group.\
                
                Provide specific recommendations including:\
                - Top 4-5 must-visit attractions\
                - Brief description of each place\
                - Approximate visit duration\
                - Entry fees (if any)\
                - Best time to visit\
                
                Focus on attractions within and close to {place}.\
                Return the response in clear text format.
                """
            ),
            markdown=True,
        )
        
        response_stream = await agent.arun(message, stream=False)
        response_text = extract_text_from_response(response_stream)
        
        if hasattr(agent, 'close') and callable(agent.close):
            try:
                if asyncio.iscoroutinefunction(agent.close):
                    await agent.close()
                else:
                    agent.close()
            except Exception as e:
                print(f"Error closing fallback agent: {e}")
        
        return response_text
        
    except Exception as e:
        print(f"Error in fallback sightseeing agent: {e}")
        return None


## location agent
async def location_mcp_agent(message: str, place: str, days_left: int, tourist_destination: str, places_visited: list):
    response_text = None
    try:
        working_cmd = await test_mcp_connection()
        if not working_cmd:
            print("No working MCP command found, using fallback agent without MCP tools")
            return await location_fallback_agent(message, place, days_left)
        
        async with MCPTools(working_cmd) as mcptools:
            
            print("MCPTools initializing for Location Agent...")
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
                instructions=dedent(f"""\
                    You are a travel agent. The tourists have {days_left} days left in their trip.\
                    Starting from {place}, recommend the next best tourist destination they should visit.\
                    Keep in mind the next place must be within {tourist_destination} and cannot repeat any places already visited: {places_visited}.\
                        
                    Consider:\
                    - Distance from {place} (not too far for the remaining days)\
                    - Popular tourist attractions\
                    - Accommodation availability\
                    - Transportation connectivity\
                    
                    Return ONLY the name of the recommended destination, nothing else.\
                    """
                ),
                tools=[mcptools],
                markdown=True,
            )
            print("Location Agent initialized!")
            
            try:
                response_stream = await agent.arun(message, stream=False)
                await apprint_run_response(response_stream, markdown=True)
                
                # extracting the actual text content
                response_text = extract_text_from_response(response_stream)
                
            finally:
                if hasattr(agent, 'close') and callable(agent.close):
                    try:
                        if asyncio.iscoroutinefunction(agent.close):
                            await agent.close()
                        else:
                            agent.close()
                        print("Agent closed")
                    except Exception as e:
                        print(f"Error closing agent: {e}")
                    
        print("MCPTools context exited.")
    
    except Exception as e:
        print(f"Error occurred in location_agent: {e}")
        response_text = await location_fallback_agent(message, place, days_left, tourist_destination)
        
    return response_text

## fallback location agent
async def location_fallback_agent(message: str, place: str, days_left: int, tourist_destination: str, places_visited: list):
    """
    Fallback location agent without MCP tools
    """
    try:
        agent = Agent(
            model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
            instructions=dedent(f"""\
                You are a travel agent. The tourists have {days_left} days left in their trip.\
                Starting from {place}, recommend the next best tourist destination they should visit.\
                Keep in mind the next place must be within {tourist_destination} and cannot repeat any places already visited: {places_visited}.\
                    
                Consider:\
                - Distance from {place} (not too far for the remaining days)\
                - Popular tourist attractions\
                - Accommodation availability\
                - Transportation connectivity\
                
                Return ONLY the name of the recommended destination, nothing else.\
                """
            ),
            markdown=True,
        )
        
        response_stream = await agent.arun(message, stream=False)
        response_text = extract_text_from_response(response_stream)
        
        if hasattr(agent, 'close') and callable(agent.close):
            try:
                if asyncio.iscoroutinefunction(agent.close):
                    await agent.close()
                else:
                    agent.close()
            except Exception as e:
                print(f"Error closing fallback agent: {e}")
        
        return response_text
        
    except Exception as e:
        print(f"Error in fallback location agent: {e}")
        return None

# function to demonstrate usage
async def test_agents():
    """
    Test function to show how all agents work and return text
    """
    print("Testing Transport Agent")
    transport_result = await transport_mcp_agent(
        "Find me the price of traveling from Kolkata to Sikkim using car, train, flight. Please return in json format.",
        people=2
    )
    print(f"Transport Result Type: {type(transport_result)}")
    print(f"Transport Result: {transport_result}")
    
    print("Testing Hotel Booking Agent")
    hotel_result = await hotel_booking_mcp_agent(
        "Find me affordable hotels in Sikkim for 2 people for 3 nights",
        place="Sikkim",
        people=2
    )
    print(f"Hotel Result Type: {type(hotel_result)}")
    print(f"Hotel Result: {hotel_result}")
    
    print("Testing Sightseeing Agent")
    sightseeing_result = await sightseeing_mcp_agent(
        "What are the top 5 sightseeing places in Sikkim?",
        place="Sikkim",
        people=2
    )
    print(f"Sightseeing Result Type: {type(sightseeing_result)}")
    print(f"Sightseeing Result: {sightseeing_result}")
    
    print("Testing Location Agent")
    location_result = await location_mcp_agent(
        "What's the next best place to visit from Sikkim for a 7-day trip?",
        place="Sikkim",
        days_left=4
    )
    print(f"Location Result Type: {type(location_result)}")
    print(f"Location Result: {location_result}")
            

if __name__ == "__main__":
    print("Starting main execution...")
    asyncio.run(test_agents())