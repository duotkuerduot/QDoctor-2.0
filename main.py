import os
import logging
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from core.orchestrator import Orchestrator
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QDoctor 2.0 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qdoctor-ai-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdoctor = Orchestrator()

class AskRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "QDoctor System Ready", "status": "healthy"}

@app.post("/ask")
async def ask_question(query: str):
    logger.info(f"Received query: {query}")
    try:
        ai_response = qdoctor.process_query(query)
        return {"response": ai_response}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)