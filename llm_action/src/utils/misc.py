from uuid import uuid4

def random_id(short: bool = False) -> str:
    return str(uuid4())[:4] if short else str(uuid4())[:8]