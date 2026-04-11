from storage.supabase_client import supabase_client
import logging

logger = logging.getLogger(__name__)


class QCache:
    def get(self, question: str):
        """Retrieve cached answer from Supabase."""
        clean_q = question.strip().lower()
        try:
            result = (
                supabase_client.table("qdoctor_cache")
                .select("answer")
                .eq("question_key", clean_q)
                .limit(1)
                .execute()
            )
            if result.data:
                return result.data[0]["answer"]
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None

    def set(self, question: str, answer: str):
        """Store validated answer in Supabase."""
        clean_q = question.strip().lower()
        try:
            supabase_client.table("qdoctor_cache").upsert(
                {"question_key": clean_q, "answer": answer},
                on_conflict="question_key"
            ).execute()
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
