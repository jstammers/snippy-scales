"""Signal conversion: signed positions → boolean entry/exit arrays.

Defines:

* :class:`SignalBundle` — a frozen dataclass holding the four boolean arrays
  (long entries, long exits, short entries, short exits) that raptorbt consumes.
* :class:`PositionInterpreter` — a :class:`~typing.Protocol` so that alternative
  interpretation rules (e.g. threshold-based, calendar-rebal-aware) can be
  swapped in without changing runner code.
* :class:`SignFlipInterpreter` — the default implementation that detects sign
  changes in a signed position array.
* :func:`positions_to_signals` — module-level convenience wrapper kept for
  backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# ── SignalBundle ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignalBundle:
    """Four boolean arrays produced by interpreting a position time series.

    Attributes:
        long_entries: ``True`` on bars where a long position should open.
        long_exits: ``True`` on bars where a long position should close.
        short_entries: ``True`` on bars where a short position should open.
        short_exits: ``True`` on bars where a short position should close.
    """

    long_entries: np.ndarray
    long_exits: np.ndarray
    short_entries: np.ndarray
    short_exits: np.ndarray


# ── PositionInterpreter Protocol ──────────────────────────────────────────────


@runtime_checkable
class PositionInterpreter(Protocol):
    """Protocol for objects that convert position arrays to :class:`SignalBundle`.

    Implement this protocol to provide alternative signal generation rules
    (e.g. percentage-threshold rebalancing, calendar-driven re-entry) without
    changing the runner infrastructure.
    """

    def interpret(self, positions: np.ndarray) -> SignalBundle:
        """Convert a signed-position array to entry/exit signal arrays.

        Args:
            positions: Float array of signed target positions. Positive → long,
                negative → short, zero → flat.

        Returns:
            :class:`SignalBundle` with four boolean arrays of the same length.
        """
        ...


# ── SignFlipInterpreter ───────────────────────────────────────────────────────


class SignFlipInterpreter:
    """Default :class:`PositionInterpreter` based on sign-change detection.

    Entries are triggered on the bar where the sign first becomes positive
    (long) or negative (short).  Exits are triggered on the bar where the sign
    changes away from positive or negative respectively.

    Direct long-to-short transitions (no flat bar) produce simultaneous long
    exit and short entry signals on the transition bar.
    """

    def interpret(self, positions: np.ndarray) -> SignalBundle:
        """Convert *positions* using sign-flip logic.

        Args:
            positions: Float array of signed target positions.

        Returns:
            :class:`SignalBundle` derived from sign-change detection.
        """
        is_long = positions > 0.0
        is_short = positions < 0.0

        prev_long = np.empty_like(is_long)
        prev_long[0] = False
        prev_long[1:] = is_long[:-1]

        prev_short = np.empty_like(is_short)
        prev_short[0] = False
        prev_short[1:] = is_short[:-1]

        return SignalBundle(
            long_entries=is_long & ~prev_long,
            long_exits=prev_long & ~is_long,
            short_entries=is_short & ~prev_short,
            short_exits=prev_short & ~is_short,
        )


# ── Module-level convenience function ────────────────────────────────────────

_DEFAULT_INTERPRETER = SignFlipInterpreter()


def positions_to_signals(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a signed-position array to boolean entry/exit arrays.

    Convenience wrapper around :class:`SignFlipInterpreter` kept for backward
    compatibility with code that calls this function directly.

    Args:
        positions: Float array of signed target positions. Positive → long,
            negative → short, zero → flat.

    Returns:
        A 4-tuple of ``(long_entries, long_exits, short_entries, short_exits)``,
        all boolean arrays of the same length as *positions*.

    Examples:
        >>> pos = np.array([0.0, 0.8, 0.9, 0.0, -0.7, 0.0])
        >>> le, lx, se, sx = positions_to_signals(pos)
        >>> le.tolist()  # long entry at index 1
        [False, True, False, False, False, False]
    """
    bundle = _DEFAULT_INTERPRETER.interpret(positions)
    return bundle.long_entries, bundle.long_exits, bundle.short_entries, bundle.short_exits
