from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv

# --- SETUP: The Brain & The Tools ---
load_dotenv()
# 1. Connect to Database
# (Ensure you have your actual database URI here)
db = SQLDatabase.from_uri(f"postgresql://{os.getenv('SUPABASE_USER')}:{os.getenv('SUPABASE_PASSWORD')}@{os.getenv('SUPABASE_HOST')}:{os.getenv('SUPABASE_PORT')}/{os.getenv('SUPABASE_DB')}")

# 2. Define the Tools
@tool
def list_tables_tool():
    """Returns a list of all tables in the database."""
    return ", ".join(db.get_usable_table_names())

@tool
def get_schema_tool(table_names: str):
    """Returns the schema (columns and types) for the specified tables.
    Input must be a comma-separated list of table names. 
    Example: 'users, orders'
    """
    return db.get_table_info(table_names.split(", "))

@tool
def execute_sql_tool(sql_query: str):
    """Executes a SQL query and returns the results.
    Input must be a valid SQL string.
    """
    try:
        return db.run(sql_query)
    except Exception as e:
        return f"Error executing SQL: {e}"

print("My gemini API key: ", os.environ.get("GEMINI_API_KEY"))
# 3. Initialize the Model with Tools
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools([list_tables_tool, get_schema_tool, execute_sql_tool])

# --- THE AGENT RUNTIME: The "While" Loop ---
def run_agent(user_query):
    # Start the conversation history
    messages = [
        SystemMessage(content="You are a helpful SQL assistant. You execute SQL queries to answer user questions. Always check the schema before writing a query."),
        HumanMessage(content=user_query)
    ]
    
    print(f"User: {user_query}")
    
    # LOOP: Keep going until the LLM stops calling tools
    while True:
        # 1. Ask the Brain (Reasoning)
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg) # Add AI thought to history
        
        # 2. Check: Did the AI want to use a tool?
        if not ai_msg.tool_calls:
            # logic: If no tool call, the agent is done. It has the answer.
            print(f"Agent Final Answer: {ai_msg.content}")
            break

        # 3. If yes, execute the tool (Acting)
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f" -> Agent is calling tool: {tool_name} with args: {tool_args}")
            
            # This is the 'if/else' block you asked about. 
            # We map the NAME (string) to the FUNCTION (code).
            tool_output = "Error: Tool not found"
            if tool_name == "list_tables_tool":
                tool_output = list_tables_tool.invoke(tool_args)
            elif tool_name == "get_schema_tool":
                tool_output = get_schema_tool.invoke(tool_args)
            elif tool_name == "execute_sql_tool":
                tool_output = execute_sql_tool.invoke(tool_args)
            
            print(f"    <- Tool Output: {str(tool_output)[:100]}...") # Truncate log for readability

            # 4. Feed the observation back to the brain
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))

# --- RUN IT ---
if __name__ == "__main__":
    run_agent("Who bought the most expensive product?")