# main.py
import os
import json
import asyncio
import logging
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from core.orchestrator import Orchestrator
from core.auth import verify_supabase_token
from storage.chat_history import (
    create_session,
    update_session_title,
    save_message,
    get_user_sessions,
    get_session_messages,
    set_active_variant,
    delete_session,
)
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QDoctor 2.0 API")
static_dir = settings.KB_PATH
if not os.path.isabs(static_dir):
    static_dir = os.path.join(settings.BASE_DIR, static_dir)
app.mount("/pdfs", StaticFiles(directory=static_dir), name="pdfs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:3000", "https://qdoctor-frontend.hf.space"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdoctor = Orchestrator()


class AskRequest(BaseModel):
    query: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    session_id: Optional[str] = None
    parent_id: Optional[str] = None
    # When true: skip saving user message, create assistant as child of parent_id
    assistant_retry: Optional[bool] = False


class RenameRequest(BaseModel):
    title: str


class SwitchVariantRequest(BaseModel):
    parent_id: str
    message_id: str


@app.get("/")
def read_root():
    return {"message": "QDoctor System Ready", "status": "healthy"}


@app.get("/sessions")
async def list_sessions(token_payload: dict = Depends(verify_supabase_token)):
    user_id = token_payload.get("sub")
    return {"sessions": get_user_sessions(user_id)}


@app.get("/sessions/{session_id}/messages")
async def list_session_messages(
    session_id: str,
    token_payload: dict = Depends(verify_supabase_token),
):
    return {"messages": get_session_messages(session_id)}


@app.delete("/sessions/{session_id}")
async def remove_session(
    session_id: str,
    token_payload: dict = Depends(verify_supabase_token),
):
    delete_session(session_id)
    return {"status": "deleted"}


@app.patch("/sessions/{session_id}")
async def rename_session(
    session_id: str,
    body: RenameRequest,
    token_payload: dict = Depends(verify_supabase_token),
):
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    update_session_title(session_id, title)
    return {"status": "renamed", "title": title}


@app.post("/sessions/{session_id}/switch-variant")
async def switch_variant(
    session_id: str,
    body: SwitchVariantRequest,
    token_payload: dict = Depends(verify_supabase_token),
):
    set_active_variant(session_id, body.parent_id, body.message_id)
    return {"messages": get_session_messages(session_id)}


@app.post("/ask")
async def ask_question(
    request: Optional[AskRequest] = None,
    query: Optional[str] = Query(default=None),
    token_payload: dict = Depends(verify_supabase_token),
):
    user_query = (request.query if request and request.query else query)
    if not user_query:
        raise HTTPException(status_code=422, detail="Missing 'query'.")
    user_id = token_payload.get("sub")
    history = request.history if request and request.history else []
    try:
        ai_response = qdoctor.process_query(
            user_query=user_query, user_id=user_id, history=history
        )
        return ai_response
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}


@app.post("/ask/stream")
async def ask_question_stream(
    request: Optional[AskRequest] = None,
    query: Optional[str] = Query(default=None),
    token_payload: dict = Depends(verify_supabase_token),
):
    user_query = (request.query if request and request.query else query)
    if not user_query:
        raise HTTPException(status_code=422, detail="Missing 'query'.")

    user_id = token_payload.get("sub")
    user_email = token_payload.get("email", "unknown")
    logger.info(f"[Stream] Query from {user_email}: {user_query}")

    history = request.history if request and request.history else []
    session_id = request.session_id if request else None
    parent_id = request.parent_id if request else None
    is_assistant_retry = request.assistant_retry if request else False

    # ── Session management ──────────────────────────────────────
    is_new_session = False
    if not session_id:
        title = user_query[:80] + ("..." if len(user_query) > 80 else "")
        session_id = create_session(user_id, title)
        is_new_session = True

    # ── Save messages based on mode ─────────────────────────────
    user_msg_id = None
    # The parent for the assistant message
    assistant_parent_id = None

    if is_assistant_retry:
        # ASSISTANT RETRY: Don't create a user message.
        # parent_id is the user message's DB ID.
        # The new assistant will be a sibling of the old assistant.
        assistant_parent_id = parent_id
        logger.info(f"[Stream] Assistant retry under user msg: {parent_id}")
    else:
        # NORMAL or USER EDIT/RETRY: Create a new user message.
        if session_id:
            user_msg_id = save_message(
                session_id, "user", user_query, parent_id=parent_id
            )
            if user_msg_id:
                fork_key = parent_id or "root"
                set_active_variant(session_id, fork_key, user_msg_id)
            assistant_parent_id = user_msg_id

    async def event_generator():
        answer_parts: List[str] = []
        assistant_sources: List[str] = []
        assistant_msg_id = None

        try:
            # 1. Session info
            if session_id:
                yield f"event: session\ndata: {json.dumps({'session_id': session_id, 'is_new': is_new_session})}\n\n"
                await asyncio.sleep(0)

            # 2. User message DB ID (only if we created one)
            if user_msg_id:
                yield f"event: msg_id\ndata: {json.dumps({'user_msg_id': user_msg_id, 'parent_id': parent_id})}\n\n"
                await asyncio.sleep(0)

            # 3. Stream LLM response
            for event in qdoctor.process_query_stream(
                user_query=user_query, user_id=user_id, history=history
            ):
                yield event
                await asyncio.sleep(0)

                # Collect full answer
                if event.startswith("event: token\n"):
                    try:
                        data_line = event.split("\n")[1]
                        token = json.loads(data_line.replace("data: ", ""))
                        answer_parts.append(token)
                    except Exception:
                        pass
                elif event.startswith("event: replace\n"):
                    try:
                        data_line = event.split("\n")[1]
                        answer_parts = [json.loads(data_line.replace("data: ", ""))]
                    except Exception:
                        pass
                elif event.startswith("event: citations\n"):
                    try:
                        data_line = event.split("\n")[1]
                        citations = json.loads(data_line.replace("data: ", ""))
                        assistant_sources = [
                            json.dumps(citation)
                            for citation in citations
                            if isinstance(citation, dict)
                        ]
                    except Exception:
                        pass
                elif event.startswith("event: response\n"):
                    try:
                        data_line = event.split("\n")[1]
                        payload = json.loads(data_line.replace("data: ", ""))
                        answer_parts = [payload.get("answer_text", "")]
                        citations = payload.get("citations", [])
                        assistant_sources = [
                            json.dumps(citation)
                            for citation in citations
                            if isinstance(citation, dict)
                        ]
                    except Exception:
                        pass

            # 4. Save assistant response
            full_answer = "".join(answer_parts)
            if session_id and full_answer and assistant_parent_id:
                assistant_msg_id = save_message(
                    session_id, "assistant", full_answer,
                    sources=assistant_sources,
                    parent_id=assistant_parent_id
                )
                if assistant_msg_id:
                    set_active_variant(
                        session_id, assistant_parent_id, assistant_msg_id
                    )

            # 5. Send assistant DB ID
            yield f"event: assistant_msg_id\ndata: {json.dumps({'assistant_msg_id': assistant_msg_id, 'parent_id': assistant_parent_id})}\n\n"
            await asyncio.sleep(0)

            # 6. Done
            yield f"event: done\ndata: {json.dumps({})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error for {user_email}: {e}")
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
