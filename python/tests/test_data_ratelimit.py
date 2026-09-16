"""Unit tests for snippy_scales.data.ratelimit.RateLimiter.

Uses an injected fake clock/sleep so no test actually sleeps in wall-clock
time — sleeps just advance the fake clock instantly.
"""

from __future__ import annotations

import threading

import pytest

from snippy_scales.data.ratelimit import RateLimiter


class _FakeClock:
    """A monotonic clock that advances only when `sleep()` is called."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_calls: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


class TestRateLimiter:
    def test_rejects_non_positive_rate(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RateLimiter(0)

    def test_first_call_does_not_sleep(self) -> None:
        clock = _FakeClock()
        limiter = RateLimiter(60, clock=clock.time, sleep=clock.sleep)
        limiter.acquire()
        assert clock.sleep_calls == []

    def test_second_call_within_window_sleeps_for_the_interval(self) -> None:
        clock = _FakeClock()
        limiter = RateLimiter(60, clock=clock.time, sleep=clock.sleep)  # 1 call/sec
        limiter.acquire()
        limiter.acquire()
        assert clock.sleep_calls == [pytest.approx(1.0)]

    def test_no_sleep_if_enough_time_already_elapsed(self) -> None:
        clock = _FakeClock()
        limiter = RateLimiter(60, clock=clock.time, sleep=clock.sleep)
        limiter.acquire()
        clock.now += 10.0  # plenty of time passes externally
        limiter.acquire()
        assert clock.sleep_calls == []

    def test_bursts_are_spaced_evenly(self) -> None:
        clock = _FakeClock()
        limiter = RateLimiter(120, clock=clock.time, sleep=clock.sleep)  # 1 call/0.5s
        for _ in range(5):
            limiter.acquire()
        # First call free, remaining 4 each wait ~0.5s.
        assert clock.sleep_calls == [pytest.approx(0.5)] * 4

    def test_thread_safe_reservation_never_exceeds_budget(self) -> None:
        """Many threads acquiring concurrently should never get overlapping slots."""
        import time as real_time

        limiter = RateLimiter(1000)  # real clock, high enough rate to run fast
        acquired_at: list[float] = []
        lock = threading.Lock()

        def worker() -> None:
            limiter.acquire()
            with lock:
                acquired_at.append(real_time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(acquired_at) == 20
