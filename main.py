import os
import logging
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from core.orchestrator import Orchestrator
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QDoctor 2.0 API")

# 1. Credentials from Environment
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# 2. CORS - Updated for your Vercel Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qdoctor-ai-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Global Instances & Memory
qdoctor = Orchestrator()
chat_memory: Dict[int, List[Dict[str, str]]] = {}
MAX_MEMORY = 10 

# --- Helper Functions ---

async def send_telegram_action(chat_id: int, action: str = "typing"):
    """Sends a 'typing' status to Telegram."""
    async with httpx.AsyncClient() as client:
        await client.post(f"{TELEGRAM_API}/sendChatAction", json={"chat_id": chat_id, "action": action})

async def send_telegram_message(chat_id: int, text: str):
    """Sends the final AI response to Telegram."""
    async with httpx.AsyncClient() as client:
        try:
            await client.post(f"{TELEGRAM_API}/sendMessage", json={
                "chat_id": chat_id, 
                "text": text,
                "parse_mode": "Markdown" # Allows bolding/lists in answers
            })
        except Exception as e:
            logger.error(f"Telegram Send Error: {e}")

def get_contextual_query(chat_id: int, current_query: str) -> str:
    """Combines history into a single prompt for context-aware answers."""
    history = chat_memory.get(chat_id, [])
    if not history:
        return current_query
    
    # Simple prompt injection of history
    formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    return f"Previous Conversation:\n{formatted_history}\n\nCurrent Question: {current_query}"

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "QDoctor System Ready", "status": "healthy"}

@app.post("/ask")
async def ask_question(query: str):
    """Stateless endpoint for React frontend."""
    try:
        response = qdoctor.process_query(query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/telegram_webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """Safe async endpoint that avoids Telegram timeouts and JSON crashes."""
    try:
        body = await request.body()
        if not body:
            logger.info("Received empty webhook body.")
            return {"ok": True}
        try:
            data = await request.json()
        except Exception:
            logger.warning("Invalid JSON received.")
            return {"ok": True}
        if "message" in data and "text" in data["message"]:
            chat_id = data["message"]["chat"]["id"]
            user_text = data["message"]["text"]
            background_tasks.add_task(send_telegram_action, chat_id)
            background_tasks.add_task(process_telegram_ai, chat_id, user_text)

        return {"ok": True}

    except Exception as e:
        logger.error(f"Webhook Fatal Error: {e}")
        return {"ok": True} 

async def process_telegram_ai(chat_id: int, user_text: str):
    """Handles memory updates and Orchestrator call."""
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

        # Trim memory to keep context window clean
        if len(chat_memory[chat_id]) > MAX_MEMORY:
            chat_memory[chat_id] = chat_memory[chat_id][-MAX_MEMORY:]

        # Send response
        await send_telegram_message(chat_id, ai_response)

    except Exception as e:
        logger.error(f"Processing Error: {e}")
        await send_telegram_message(chat_id, "I'm sorry, I'm having trouble processing that right now.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
