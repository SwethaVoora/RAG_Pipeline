from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import HTTPException
from agent_graph import run_langgraph_agent_query
import uvicorn

app = FastAPI()

# Pydantic model for input validation
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_db(request: QueryRequest):
    try:
        response = run_langgraph_agent_query(request.question)
        return {
            "answer": response["answer"],
            "agent": "langgraph",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangGraph agent execution failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
