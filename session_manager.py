import secrets

def create_session_id() -> str:
    """Generate an unpredictable identifier for one visitor session."""
    return secrets.token_urlsafe(32)