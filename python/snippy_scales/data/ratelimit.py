"""A simple, thread-safe rate limiter for provider clients with per-minute quotas.

Alpaca's free tier caps historical API calls at 200/minute (as of 2026), and
the ``alpaca-py`` SDK only reacts to a 429 *after* it happens — it does not
throttle proactively. :class:`RateLimiter` sits in front of every outbound
request so a multi-threaded backfill stays under quota instead of tripping it.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class RateLimiter:
    """Spaces calls to :meth:`acquire` evenly across a per-minute budget.

    Uses a reservation scheme rather than a token bucket: each call reserves
    the next available slot under lock, then sleeps (without holding the
    lock) until that slot arrives. This lets multiple threads queue up
    without serialising on the sleep itself, while still guaranteeing no more
    than ``calls_per_minute`` requests start in any rolling 60-second window.

    Args:
        calls_per_minute: Maximum number of :meth:`acquire` calls allowed to
            *start* per 60-second window. Must be positive.
        clock: Monotonic time source, injectable for deterministic tests.
        sleep: Sleep function, injectable for deterministic tests.
    """

    def __init__(
        self,
        calls_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if calls_per_minute <= 0:
            raise ValueError(f"calls_per_minute must be positive, got {calls_per_minute}.")
        self._interval = 60.0 / calls_per_minute
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_available = clock()

    def acquire(self) -> None:
        """Block the calling thread until its reserved slot arrives."""
        with self._lock:
            now = self._clock()
            start_at = max(self._next_available, now)
            self._next_available = start_at + self._interval
        wait = start_at - now
        if wait > 0:
            self._sleep(wait)
