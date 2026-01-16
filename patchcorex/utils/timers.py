import time
from contextlib import contextmanager
from typing import Dict, Iterator


@contextmanager
def timer(store: Dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    store[key] = time.perf_counter() - start