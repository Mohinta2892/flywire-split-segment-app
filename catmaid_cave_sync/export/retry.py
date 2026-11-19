"""Transient HTTP retry with exponential backoff."""
import time
from urllib.parse import urlparse


def extract_host(exc: Exception) -> str:
    txt = str(exc)
    idx = txt.find("http")
    if idx != -1:
        end = idx
        while end < len(txt) and not txt[end].isspace() and txt[end] not in ("'", '"', ")"):
            end += 1
        host = urlparse(txt[idx:end]).netloc
        if host:
            return host
    return ""


def retry_call(fn, stage: str, max_attempts: int = 4, base_delay: float = 1.5, progress_callback=None):
    """Retry transient HTTP 429/502/503/504 or connection errors with exponential backoff."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_transient = any(
                code in msg
                for code in ("503", "502", "504", "429", "temporarily unavailable", "bad gateway", "timed out", "connection reset")
            )
            if not is_transient or attempt == max_attempts:
                host = extract_host(exc)
                where = f"{stage} ({host})" if host else stage
                raise RuntimeError(f"[{where}] {exc}") from exc
            wait_s = base_delay * (2 ** (attempt - 1))
            if progress_callback:
                progress_callback(
                    f"Server temporarily busy during {stage} (attempt {attempt}/{max_attempts}); retrying in {wait_s:.1f}s..."
                )
            time.sleep(wait_s)
    raise last_exc