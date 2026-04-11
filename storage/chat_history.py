# storage/chat_history.py
import json
import logging
from typing import Optional, List, Dict, Any
from storage.supabase_client import supabase_client

logger = logging.getLogger(__name__)


def _deserialize_sources(sources: Optional[List[Any]]) -> List[Any]:
    parsed_sources: List[Any] = []
    for item in sources or []:
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
                parsed_sources.append(parsed)
                continue
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed_sources.append(item)
                continue
        parsed_sources.append(item)
    return parsed_sources


# ─── Session CRUD ───────────────────────────────────────────────

def create_session(user_id: str, title: str = "New Chat") -> Optional[str]:
    try:
        result = (
            supabase_client.table("chat_sessions")
            .insert({"user_id": user_id, "title": title, "active_variants": {}})
            .execute()
        )
        return result.data[0]["id"] if result.data else None
    except Exception as e:
        logger.warning(f"Failed to create session: {e}")
        return None


def update_session_title(session_id: str, title: str):
    try:
        supabase_client.table("chat_sessions").update(
            {"title": title}
        ).eq("id", session_id).execute()
    except Exception as e:
        logger.warning(f"Failed to update session title: {e}")


def get_user_sessions(user_id: str, limit: int = 50) -> List[Dict]:
    try:
        result = (
            supabase_client.table("chat_sessions")
            .select("id, title, created_at, updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.warning(f"Failed to get sessions: {e}")
        return []


def delete_session(session_id: str):
    try:
        supabase_client.table("chat_sessions").delete().eq(
            "id", session_id
        ).execute()
    except Exception as e:
        logger.warning(f"Failed to delete session: {e}")


# ─── Message operations ────────────────────────────────────────

def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[str]] = None,
    parent_id: Optional[str] = None,
) -> Optional[str]:
    """Saves a message and returns its ID."""
    try:
        row: Dict[str, Any] = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "sources": sources or [],
        }
        if parent_id:
            row["parent_id"] = parent_id

        result = supabase_client.table("chat_messages").insert(row).execute()
        return result.data[0]["id"] if result.data else None
    except Exception as e:
        logger.warning(f"Failed to save message: {e}")
        return None


def get_all_session_messages(session_id: str) -> List[Dict]:
    """Returns ALL messages in a session (full tree)."""
    try:
        result = (
            supabase_client.table("chat_messages")
            .select("id, role, content, sources, parent_id, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        messages = result.data or []
        for message in messages:
            message["sources"] = _deserialize_sources(message.get("sources"))
        return messages
    except Exception as e:
        logger.warning(f"Failed to get messages: {e}")
        return []


def get_session_active_variants(session_id: str) -> Dict[str, str]:
    try:
        result = (
            supabase_client.table("chat_sessions")
            .select("active_variants")
            .eq("id", session_id)
            .single()
            .execute()
        )
        return result.data.get("active_variants") or {} if result.data else {}
    except Exception as e:
        logger.warning(f"Failed to get active variants: {e}")
        return {}


def set_active_variant(session_id: str, parent_key: str, chosen_message_id: str):
    """
    Sets which child is active at a fork point.
    parent_key: the parent message's UUID, or "root" for root-level forks.
    """
    try:
        current = get_session_active_variants(session_id)
        current[parent_key] = chosen_message_id
        supabase_client.table("chat_sessions").update(
            {"active_variants": current}
        ).eq("id", session_id).execute()
    except Exception as e:
        logger.warning(f"Failed to set active variant: {e}")


# ─── Tree traversal ────────────────────────────────────────────

def build_active_path(session_id: str) -> List[Dict]:
    """
    Walks the message tree following active_variants at each fork.
    Returns flat list with _variants and _active_variant metadata.
    """
    all_messages = get_all_session_messages(session_id)
    if not all_messages:
        return []

    active_variants = get_session_active_variants(session_id)

    # Build children lookup: parent_id -> list of children
    # Use "root" as key for messages with no parent
    children: Dict[str, List[Dict]] = {}
    by_id: Dict[str, Dict] = {}

    for m in all_messages:
        by_id[m["id"]] = m
        parent_key = m.get("parent_id") or "root"
        children.setdefault(parent_key, []).append(m)

    # Walk from root
    path: List[Dict] = []
    current_key = "root"

    while current_key in children:
        kids = children[current_key]
        if not kids:
            break

        if len(kids) == 1:
            chosen = kids[0]
        else:
            # Fork point — check which variant is active
            active_id = active_variants.get(current_key)
            chosen = None
            if active_id:
                # Find the active child
                for k in kids:
                    if k["id"] == active_id:
                        chosen = k
                        break
            if not chosen:
                # Default to most recent
                chosen = kids[-1]

        # Attach variant metadata if this is a fork
        if len(kids) > 1:
            variant_list = [
                {
                    "id": k["id"],
                    "content": k["content"],
                    "sources": k.get("sources") or [],
                }
                for k in kids
            ]
            chosen_index = next(
                (i for i, k in enumerate(kids) if k["id"] == chosen["id"]), 0
            )
            chosen = {
                **chosen,
                "_variants": variant_list,
                "_active_variant": chosen_index,
            }

        path.append(chosen)
        current_key = chosen["id"]

    return path


def get_session_messages(session_id: str) -> List[Dict]:
    """Public API: returns active path with variant metadata."""
    return build_active_path(session_id)


# ─── Legacy ─────────────────────────────────────────────────────

def save_chat_message(
    user_id: str, user_query: str, ai_response: str, intent: str = "MENTAL_HEALTH"
):
    try:
        supabase_client.table("chat_history").insert({
            "user_id": user_id,
            "user_query": user_query,
            "ai_response": ai_response,
            "intent": intent,
        }).execute()
    except Exception as e:
        logger.warning(f"Failed to save chat history: {e}")
