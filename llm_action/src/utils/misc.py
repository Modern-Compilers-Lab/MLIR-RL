from uuid import uuid4

def random_id(short: bool = True) -> str:
    return str(uuid4())[:4] if short else str(uuid4())[:8]