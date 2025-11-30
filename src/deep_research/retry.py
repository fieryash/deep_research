"""Retry helpers for wrapping LLM calls."""
from __future__ import annotations

import logging
from typing import Any, Dict

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

_DEFAULT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_KEYWORDS = ("rate limit", "quota", "exceeded", "temporarily unavailable")


class QuotaExhaustedError(RuntimeError):
    """Raised when a provider reports zero remaining quota."""


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response:
        for attr in ("status", "status_code", "code"):
            value = getattr(response, attr, None)
            if isinstance(value, int):
                return value
    return None


def _is_zero_quota(exc: Exception) -> bool:
    """Detect cases where the provider reports no available quota."""

    message = str(exc).lower()
    if "insufficient_quota" in message or "insufficient quota" in message:
        return True
    return "limit: 0" in message or ("quota exceeded" in message and "limit" in message)


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, QuotaExhaustedError):
        return False
    if _is_zero_quota(exc):
        # Failing fast avoids hammering a provider when quota is disabled
        return False

    status_code = _extract_status_code(exc)
    if status_code and status_code in _DEFAULT_STATUS_CODES:
        return True

    name = exc.__class__.__name__.lower()
    if any(keyword in name for keyword in ("ratelimit", "resourceexhausted", "quota")):
        return True

    message = str(exc).lower()
    return any(keyword in message for keyword in _KEYWORDS)


async def ainvoke_with_retry(
    chain: Any,
    payload: Dict[str, Any],
    *,
    attempts: int = 3,
    base: float = 0.5,
    max_wait: float = 6.0,
):
    """Call `chain.ainvoke` with exponential backoff for transient errors."""

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_random_exponential(multiplier=base, max=max_wait),
        retry=retry_if_exception(_should_retry),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    ):
        with attempt:
            try:
                return await chain.ainvoke(payload)
            except Exception as exc:
                if _is_zero_quota(exc):
                    raise QuotaExhaustedError(
                        "Provider reported zero available quota. Switch models or enable billing."
                    ) from exc
                raise


__all__ = ["QuotaExhaustedError", "ainvoke_with_retry"]
