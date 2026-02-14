import os
import logging
import httpx
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel # Added for /docs compatibility
from typing import List, Dict, Optional
from core.orchestrator import Orchestrator
from config.settings import settings

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QDoctor 2.0 API")

# --- Configuration & Global State ---
BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN
TELEGRAM_API = f"https://149.154.167.220/bot{BOT_TOKEN}"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qdoctor-ai-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI & Memory
qdoctor = Orchestrator()
chat_memory: Dict[int, List[Dict[str, str]]] = {}
MAX_MEMORY = 10 

class AskRequest(BaseModel):
    query: str

# --- Helper Functions ---

async def send_telegram_action(chat_id: int, action: str = "typing"):
    """Sends a 'typing' status to Telegram."""
    if not BOT_TOKEN:
        return

    # Use verify=False because the SSL certificate is issued for 'api.telegram.org', not the IP address.
    async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
        try:
            await client.post(
                f"{TELEGRAM_API}/sendChatAction",
                json={"chat_id": chat_id, "action": action},
            )
        except Exception as e:
            logger.error(f"Telegram Action Error: {e}")

async def send_telegram_message(chat_id: int, text: str):
    """Sends the final AI response to Telegram."""
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN missing.")
        return

    async with httpx.AsyncClient(verify=False, timeout=20.0) as client:
        try:
            response = await client.post(f"{TELEGRAM_API}/sendMessage", json={
                "chat_id": chat_id, 
                "text": text,
                "parse_mode": "Markdown" 
            })
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Telegram Send Error: {e}")

def get_contextual_query(chat_id: int, current_query: str) -> str:
    history = chat_memory.get(chat_id, [])
    if not history:
        return current_query
    
    formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    return f"Previous Conversation:\n{formatted_history}\n\nCurrent Question: {current_query}"

async def process_telegram_ai(chat_id: int, user_text: str):
    try:
        full_query = get_contextual_query(chat_id, user_text)
        ai_response = qdoctor.process_query(full_query)

        # Update Memory
        if chat_id not in chat_memory:
            chat_memory[chat_id] = []
        chat_memory[chat_id].append({"role": "user", "content": user_text})
        chat_memory[chat_id].append({"role": "assistant", "content": ai_response})

        if len(chat_memory[chat_id]) > MAX_MEMORY:
            chat_memory[chat_id] = chat_memory[chat_id][-MAX_MEMORY:]

        await send_telegram_message(chat_id, ai_response)
    except Exception as e:
        logger.error(f"AI Processing Error: {e}")

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "QDoctor System Ready", "status": "healthy"}

@app.post("/ask")
async def ask_question(request_data: AskRequest):
    """
    Standard endpoint for manual testing via /docs.
    Takes JSON: {"query": "your question here"}
    """
    try:
        response = qdoctor.process_query(request_data.query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/telegram_webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        message = data.get("message") or data.get("edited_message")
        
        if message and "text" in message:
            chat_id = message["chat"]["id"]
            user_text = message["text"]
            
            background_tasks.add_task(send_telegram_action, chat_id)
            background_tasks.add_task(process_telegram_ai, chat_id, user_text)

        return {"ok": True}
    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        return {"ok": True} 

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)