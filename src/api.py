from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.main import search_songs, generate_response

app = FastAPI(title="Music RAG API", description="API for suggesting songs based on user input")

# Define the data input format
class Query(BaseModel):
    text: str
    limit: int = Field(default=5, ge=1, le=20)

# Define API endpoints
@app.get("/")
def home():
    return {"message": "Music RAG API is running"}

# Endpoint to get song recommendations based on user query
@app.post("/recommend")
async def get_recommendation (query: Query):
    try:
        results = search_songs(query.text, limit=query.limit)
        answer = generate_response(query.text, results)

        return {"query": query.text,
                "recommendation": answer,
                "sources": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))