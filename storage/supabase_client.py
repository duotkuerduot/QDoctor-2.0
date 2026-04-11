from supabase import create_client, Client
from config.settings import settings


def get_supabase_client() -> Client:
    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_KEY  # Service role — backend only
    )


supabase_client = get_supabase_client()
