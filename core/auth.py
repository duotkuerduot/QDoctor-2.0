import logging
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from storage.supabase_client import supabase_client

logger = logging.getLogger(__name__)
bearer_scheme = HTTPBearer(auto_error=False)


def verify_supabase_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> dict:
    """
    Validates the Supabase access token by asking Supabase Auth.
    Returns a normalized payload (sub/email) used by route handlers.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token. Please log in again."
        )

    token = credentials.credentials
    try:
        response = supabase_client.auth.get_user(token)
        user = response.user if response else None
        if not user:
            raise ValueError("No user returned from Supabase Auth")

        return {
            "sub": str(user.id),
            "email": user.email,
        }
    except Exception as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token. Please log in again."
        )
