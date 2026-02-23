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
SYSTEM_PROMPT = """You are an expert PostgreSQL agent.
BUSINESS RULES:
- If asked about a product's catalog price, ALWAYS use products.price.
- If asked about revenue or how much a user paid, ALWAYS use order_items.price_at_purchase.
- For any user-level analysis, user identity MUST be users.id (NOT users.name).
- users.name is only a display label and can have duplicates.
- If asked about details about a user's orders by specifying a name, ALWAYS check if there are multiple
    users with that same name. If yes, then send a response statingthat there are multiple users with the same name and hence cannot be used for analysis.
Always check schemas before querying."""

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
    normalized = sql_query.strip().lower()
    if not normalized.startswith("select"):
        return "Error: only read-only SELECT queries are allowed."
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


def _extract_message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts = [part.get("text", "") for part in message.content if isinstance(part, dict)]
        return "\n".join(p for p in parts if p)
    return str(message.content)


def run_langgraph_agent_query(question: str) -> dict:
    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
    }
    final_answer = ""

    for event in agent_app.stream(initial_state, config={"recursion_limit": 20}):
        for key, value in event.items():
            print(f"--- Node: {key} ---")

            messages = value.get("messages", [])
            if not messages:
                continue

            if key == "agent":
                ai_msg = messages[-1]
                tool_calls = getattr(ai_msg, "tool_calls", []) or []
                if not tool_calls:
                    final_answer = _extract_message_text(ai_msg)

            if key == "tools":
                for msg in messages:
                    tool_name = getattr(msg, "name", None)
                    if tool_name:
                        print(f"    [Action] Executed Tool: {tool_name}")

    return {"answer": final_answer}

# --- 6. RUN IT ---
if __name__ == "__main__":
    # query = "Identify the product that has generated the highest total revenue. Then, analyze the users who bought it—are they one-time shoppers, or do they tend to buy multiple items?"
    query = "Calculate the total lifetime revenue for the user Charlie Williams."
    print(f"User: {query}\n")
    output = run_langgraph_agent_query(query)
    print("\n" + "=" * 50)
    print("FINAL BUSINESS INSIGHT:")
    print("=" * 50)
    print(output["answer"])