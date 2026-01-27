from __future__ import annotations

from hello_world import hello_world

def get_hello_message() -> str:
    # Business logic lives here (simple now, scalable later)
    return hello_world()
