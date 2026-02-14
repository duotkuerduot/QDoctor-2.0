import os
import logging
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from core.orchestrator import Orchestrator
from config.settings import settings
import uvicorn

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QDoctor 2.0 API")

TELEGRAM_API = settings.TELEGRAM_API
BOT_TOKEN = settings.TELEGRAM_BOT_TOKEN

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

# --- Helper Functions ---

async def send_telegram_action(chat_id: int, action: str = "typing"):
    """Sends a 'typing' status to Telegram."""
    if not BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN missing in settings. Skipping action.")
        return

    # Use a new client for each request to avoid event loop issues in background tasks
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(
                f"{TELEGRAM_API}/sendChatAction",
                json={"chat_id": chat_id, "action": action},
            )
        except httpx.HTTPError as e:
            logger.error(f"Telegram Action Error: {e}")

async def send_telegram_message(chat_id: int, text: str):
    """Sends the final AI response to Telegram."""
    if not BOT_TOKEN:
        logger.error("Cannot send Telegram message: TELEGRAM_BOT_TOKEN is missing.")
        return

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            await client.post(f"{TELEGRAM_API}/sendMessage", json={
                "chat_id": chat_id, 
                "text": text,
                "parse_mode": "Markdown" 
            })
        except httpx.HTTPError as e:
            logger.error(f"Telegram Send Error: {e}")

def get_contextual_query(chat_id: int, current_query: str) -> str:
    """Combines history into a single prompt for context-aware answers."""
    history = chat_memory.get(chat_id, [])
    if not history:
        return current_query
    
    formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    return f"Previous Conversation:\n{formatted_history}\n\nCurrent Question: {current_query}"

async def process_telegram_ai(chat_id: int, user_text: str):
    """Background task: Handles memory updates and Orchestrator call."""
    try:
        # Prepare Context
        full_query = get_contextual_query(chat_id, user_text)
        
        # Call AI 
        ai_response = qdoctor.process_query(full_query)

        # Update Memory
        if chat_id not in chat_memory:
            chat_memory[chat_id] = []
        
        chat_memory[chat_id].append({"role": "user", "content": user_text})
        chat_memory[chat_id].append({"role": "assistant", "content": ai_response})

        # Trim memory
        if len(chat_memory[chat_id]) > MAX_MEMORY:
            chat_memory[chat_id] = chat_memory[chat_id][-MAX_MEMORY:]

        # Send response
        await send_telegram_message(chat_id, ai_response)

    except Exception as e:
        logger.error(f"Processing Error: {e}")
        await send_telegram_message(chat_id, "I'm sorry, I encountered an internal error. Please try again.")

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "QDoctor System Ready", "status": "healthy"}

@app.post("/ask")
async def ask_question(request: Request):
    """Stateless endpoint for React frontend."""
    # FIX: Updated to read JSON body properly for frontend compatibility
    try:
        body = await request.json()
        query = body.get("query") if isinstance(body, dict) else str(body)
        
        if not query:
            return {"error": "Query parameter is missing"}

        response = qdoctor.process_query(query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/telegram_webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """Safe async endpoint that avoids Telegram timeouts."""
    try:
        body = await request.body()
        if not body:
            return {"ok": True}
        
        try:
            data = await request.json()
        except Exception:
            return {"ok": True}

        # Check for Message vs Edited Message
        message = data.get("message") or data.get("edited_message")
        
        if message and "text" in message:
            chat_id = message["chat"]["id"]
            user_text = message["text"]
            
            # Send 'typing' immediately
            background_tasks.add_task(send_telegram_action, chat_id)
            # Process AI in background
            background_tasks.add_task(process_telegram_ai, chat_id, user_text)

        return {"ok": True}

    except Exception as e:
        logger.error(f"Webhook Fatal Error: {e}")
        return {"ok": True} 

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)