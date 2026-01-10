from uuid import uuid4

def random_id() -> str:
    return str(uuid4())[:8]