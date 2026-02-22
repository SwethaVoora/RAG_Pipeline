import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langgraph.graph.message import add_messages

# LangGraph Imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# --- 1. SETUP TOOLS (Same as before) ---
db = SQLDatabase.from_uri(f"postgresql://{os.getenv('SUPABASE_USER')}:{os.getenv('SUPABASE_PASSWORD')}@{os.getenv('SUPABASE_HOST')}:{os.getenv('SUPABASE_PORT')}/{os.getenv('SUPABASE_DB')}")

@tool
def list_tables_tool():
    """List all table names in the database."""
    return ", ".join(db.get_usable_table_names())

@tool
def get_schema_tool(table_names: str):
    """Get the schema for specific tables. Input: comma-separated list of tables."""
    return db.get_table_info(table_names.split(", "))

@tool
def execute_sql_tool(sql_query: str):
    """Execute a SQL query. Input: valid SQL string."""
    try:
        return db.run(sql_query)
    except Exception as e:
        return f"Error: {e}"

tools = [list_tables_tool, get_schema_tool, execute_sql_tool]

# --- 2. SETUP MODEL ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# --- 3. DEFINE STATE ---
# The "State" is the memory of the agent. It just holds the list of messages.
class AgentState(TypedDict):
    # messages: list[BaseMessage]
    messages: Annotated[list[BaseMessage], add_messages]

# --- 4. DEFINE NODES ---

def call_model(state: AgentState):
    """The 'Brain' Node: Just calls the LLM."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    # We return a dictionary to UPDATE the state
    return {"messages": [response]}

def should_continue(state: AgentState) -> Literal["tools", END]:
    """The 'Logic' Edge: Decides where to go next."""
    last_message = state['messages'][-1]
    
    # If the LLM has made a tool call, go to the 'tools' node
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, stop.
    return END

# --- 5. BUILD THE GRAPH ---

workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools)) # LangGraph has a pre-built node for running tools!

# Define the entry point (Where do we start?)
workflow.set_entry_point("agent")

# Add the edges (The Logic)
workflow.add_conditional_edges(
    "agent",            # Start at Agent
    should_continue,    # Run this logic function
    ["tools", END]      # Map the output to these nodes
)

workflow.add_edge("tools", "agent") # Loop back: Tools -> Agent

# Compile the graph
agent_app = workflow.compile()

# --- 6. RUN IT ---
# --- 6. RUN IT ---
if __name__ == "__main__":
    query = "Identify the product that has generated the highest total revenue. Then, analyze the users who bought it—are they one-time shoppers, or do they tend to buy multiple items?"
    print(f"User: {query}\n")
    
    initial_state = {
        "messages": [
            SystemMessage(content="""You are an expert PostgreSQL agent.
BUSINESS RULES:
- If asked about a product's catalog price, ALWAYS use products.price.
- If asked about revenue or how much a user paid, ALWAYS use order_items.price_at_purchase.
Always check schemas before querying."""),
            HumanMessage(content=query)
        ]
    }
    
    final_answer = ""
    
    # Stream the steps so you can see what's happening
    for event in agent_app.stream(initial_state):
        for key, value in event.items():
            print(f"\n--- Node: {key} ---")
            
            # 1. Log the specific tools being executed
            if key == "tools":
                for msg in value['messages']:
                    print(f"    [Action] Executed Tool: {msg.name}")
            
            # 2. Check if the Agent has reached its final conclusion
            if key == "agent":
                ai_msg = value['messages'][-1]
                if not ai_msg.tool_calls: # If no tools are called, it has the answer
                    # Clean the output dictionary to just get the text
                    if isinstance(ai_msg.content, list):
                        final_answer = ai_msg.content[0]['text']
                    else:
                        final_answer = ai_msg.content

    print("\n" + "="*50)
    print("FINAL BUSINESS INSIGHT:")
    print("="*50)
    print(final_answer)
# if __name__ == "__main__":
#     query = "Identify the product that has generated the highest total revenue. Then, analyze the users who bought it—are they one-time shoppers, or do they tend to buy multiple items?"
#     print(f"User: {query}\n")
    
#     initial_state = {
#         "messages": [
#             SystemMessage(content="You are a SQL expert. Always check schemas before querying."),
#             HumanMessage(content=query)
#         ]
#     }
    
#     # Stream the steps so you can see what's happening
#     for event in agent_app.stream(initial_state):
#         for key, value in event.items():
#             print(f"--- Node: {key} ---")
#             # print(value) # Uncomment to see full state details