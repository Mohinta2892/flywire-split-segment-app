"""Retry helper with exponential backoff for transient CATMAID and CAVE HTTP errors."""
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def _notify(progress_callback: Callable | None, msg: str) -> None:
    if not progress_callback:
        return
    try:
        progress_callback(msg, None)
    except TypeError:
        progress_callback(msg)


def retry_call(
    fn: Callable[[], T],
    *,
    retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 12.0,
    stage: str = "API request",
    progress_callback: Callable | None = None,
) -> T:
    """Execute fn() with exponential backoff on transient network/server errors."""
    last_exc: Exception | None = None
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            delay = min(float(max_delay), float(base_delay) * (2 ** (attempt - 1)))
            _notify(
                progress_callback,
                f"{stage}: server busy, retrying ({attempt}/{attempts}) in {delay:.1f}s...",
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]