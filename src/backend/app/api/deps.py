"""app.api.deps

Shared FastAPI dependencies used across routers/endpoints.

This module is a home for small DI helpers (request metadata, auth, settings,
DB sessions, etc.). Keep dependencies side effect free so they’re safe to import.
"""


def get_request_source() -> str:
    # Example dependency; replace later with real DI (auth/user/db/etc.)
    return "api"
