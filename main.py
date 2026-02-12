import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.orchestrator import Orchestrator
import uvicorn

app = FastAPI(title="QDoctor 2.0 API")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://qdoctor-ai-frontend.vercel.app", 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize System
qdoctor = Orchestrator()

@app.get("/")
def read_root():
    return {"message": "QDoctor System Ready", "status": "healthy"}

@app.post("/ask")
def ask_question(query: str):
    try:
        response = qdoctor.process_query(query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Hugging Face Spaces requires port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)