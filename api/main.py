# This will be used to connect to our supabase database
# This will be used to create a connection to our database
# This will be used to define the api endpoint, to which the end user's query will be passed
# This will be used to define, what happens once the api endpoint is hit
# The api endpoint will make a call to the nl2sql chain and generate the corresponding SQL query
# The generated SQL query is then run on our database

from typing import Union
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException
import psycopg2
from nl2sql import nl2sql_chain  # Import your NL2SQL chain
import os
import uvicorn

sys.path.insert(0, 'RAG_Pipeline')
app = FastAPI()

# Pydantic model for input validation
class QueryRequest(BaseModel):
    question: str

# Establish database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("SUPABASE_HOST"),
        dbname=os.getenv("SUPABASE_DB"),
        user=os.getenv("SUPABASE_USER"),
        password=os.getenv("SUPABASE_PASSWORD"),
        port=os.getenv("SUPABASE_PORT", 5432)
    )

@app.post("/query")
async def query_db(request: QueryRequest):
    # Generate SQL query from the natural language input
    try:
        sql_query = nl2sql_chain.invoke(request.question)
        print("SQL QUERY Testing: ",sql_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating SQL query.")
    
    # Validate that the SQL query is read-only
    if not sql_query.strip().lower().startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")

    # Execute the SQL query on Supabase
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(sql_query)
        records = cur.fetchall()
        # Optionally, get column names from cur.description
        columns = [desc[0] for desc in cur.description]
        result = [dict(zip(columns, row)) for row in records]
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error executing SQL query.")

    return {"sql": sql_query, "result": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
