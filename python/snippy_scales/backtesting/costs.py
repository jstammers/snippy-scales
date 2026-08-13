"""Transaction-cost models for continuous-position backtests.

Costs are expressed as a **fraction of portfolio equity per bar**, so they
subtract directly from the strategy's period return:

.. code-block:: text

    net_t = pos_{t-1} * ret_t - cost_t

The *traded* quantity passed to a cost model is ``|pos_t - pos_{t-1}|``, i.e.
the change in notional exposure measured in units of equity (so ``1.0`` means
turning over the whole portfolio once).

Two implementations are provided:

* :class:`ProportionalCost` — cost is a flat fraction of traded notional.
  Mirrors the ``fees`` / ``slippage`` semantics of
  :func:`~snippy_scales.backtesting.engine.make_config`, which makes it the
  right choice when cross-checking against ``RaptorExecutionEngine``.
* :class:`FuturesCostModel` — per-contract commission plus a half-spread
  quoted in ticks.  This is the realistic model for the CME futures the
  ingestion layer targets, and it captures the fact that commission does *not*
  scale with contract notional while the bid-ask spread does.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CostModel",
    "ProportionalCost",
    "FuturesCostModel",
]


# ── CostModel Protocol ────────────────────────────────────────────────────────


@runtime_checkable
class CostModel(Protocol):
    """Protocol for transaction-cost models.

    Implement this protocol to supply an alternative cost structure (e.g.
    square-root market impact, tiered commissions, maker rebates) without
    changing the execution engine.
    """

    def cost_fraction(self, traded: np.ndarray, price: np.ndarray) -> np.ndarray:
        """Return the per-bar cost as a fraction of portfolio equity.

        Args:
            traded: Absolute change in notional exposure per bar, in units of
                equity.  ``1.0`` means the entire portfolio was turned over.
            price: Instrument price on the bar the trade executes at.

        Returns:
            Non-negative float array of the same shape as *traded*.
        """
        ...


# ── ProportionalCost ──────────────────────────────────────────────────────────


class ProportionalCost:
    """Cost proportional to traded notional.

    This reproduces the semantics of ``fees`` and ``slippage`` in
    :func:`~snippy_scales.backtesting.engine.make_config`: both are fractions
    of the notional changing hands, and both are charged on every trade.

    Args:
        fees: Commission as a fraction of traded notional (``0.001`` = 10 bp).
        slippage: Slippage as a fraction of traded notional (``0.0005`` = 5 bp).

    Raises:
        ValueError: If either argument is negative.
    """

    def __init__(self, *, fees: float = 0.001, slippage: float = 0.0005) -> None:
        if fees < 0.0:
            raise ValueError(f"fees must be non-negative, got {fees}")
        if slippage < 0.0:
            raise ValueError(f"slippage must be non-negative, got {slippage}")
        self.fees = fees
        self.slippage = slippage

    @property
    def rate(self) -> float:
        """Combined per-unit-notional cost rate."""
        return self.fees + self.slippage

    def cost_fraction(self, traded: np.ndarray, price: np.ndarray) -> np.ndarray:
        """Return ``traded * (fees + slippage)``.

        Args:
            traded: Absolute change in notional exposure per bar.
            price: Unused; present to satisfy :class:`CostModel`.

        Returns:
            Cost as a fraction of equity, per bar.
        """
        del price  # cost does not depend on price level
        return np.asarray(traded, dtype=np.float64) * self.rate


# ── FuturesCostModel ──────────────────────────────────────────────────────────


class FuturesCostModel:
    """Per-contract commission plus a tick-denominated half-spread.

    A futures contract's notional value is ``price * multiplier``, where the
    multiplier is recovered from the tick grid as ``tick_value / tick_size``.
    Trading ``traded`` units of equity notional therefore touches
    ``traded * equity / (price * multiplier)`` contracts, each incurring

    .. code-block:: text

        commission_per_side + half_spread_ticks * tick_value

    Because commission is a flat cash amount per contract while the spread
    scales with contract size, **larger contracts are cheaper per unit of
    notional**.  This is the dominant cost consideration when choosing between
    e.g. MES and ES.

    Args:
        commission_per_side: Broker + exchange fees per contract, one side.
        half_spread_ticks: Expected cost of crossing, in ticks, per side.
            ``0.5`` corresponds to paying half of a one-tick-wide market.
        tick_size: Minimum price increment (e.g. ``0.25`` for ES).
        tick_value: Cash value of one tick (e.g. ``12.50`` for ES).

    Raises:
        ValueError: If any argument is negative, or if *tick_size* or
            *tick_value* is zero.
    """

    def __init__(
        self,
        *,
        commission_per_side: float = 2.0,
        half_spread_ticks: float = 0.5,
        tick_size: float = 0.25,
        tick_value: float = 12.50,
    ) -> None:
        if commission_per_side < 0.0:
            raise ValueError(f"commission_per_side must be non-negative, got {commission_per_side}")
        if half_spread_ticks < 0.0:
            raise ValueError(f"half_spread_ticks must be non-negative, got {half_spread_ticks}")
        if tick_size <= 0.0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        if tick_value <= 0.0:
            raise ValueError(f"tick_value must be positive, got {tick_value}")

        self.commission_per_side = commission_per_side
        self.half_spread_ticks = half_spread_ticks
        self.tick_size = tick_size
        self.tick_value = tick_value

    @property
    def multiplier(self) -> float:
        """Contract multiplier (cash value of a one-point price move)."""
        return self.tick_value / self.tick_size

    @property
    def cost_per_contract_side(self) -> float:
        """Total cash cost of trading one contract, one side."""
        return self.commission_per_side + self.half_spread_ticks * self.tick_value

    def contract_notional(self, price: float) -> float:
        """Return the notional value of one contract at *price*.

        Args:
            price: Instrument price.

        Returns:
            ``price * multiplier``.
        """
        return price * self.multiplier

    def cost_bp_per_side(self, price: float) -> float:
        """Return the one-side cost in basis points of contract notional.

        Convenience for cost budgeting: multiply by two for a round turn.

        Args:
            price: Instrument price.

        Returns:
            Cost in basis points (1 bp = 0.01%).
        """
        return 1e4 * self.cost_per_contract_side / self.contract_notional(price)

    def cost_fraction(self, traded: np.ndarray, price: np.ndarray) -> np.ndarray:
        """Return the per-bar cost as a fraction of equity.

        Args:
            traded: Absolute change in notional exposure per bar, in units of
                equity.
            price: Instrument price on each bar.  Non-positive prices are
                treated as missing and produce zero cost rather than a
                division blow-up.

        Returns:
            Cost as a fraction of equity, per bar.
        """
        traded_arr = np.asarray(traded, dtype=np.float64)
        price_arr = np.asarray(price, dtype=np.float64)

        notional = price_arr * self.multiplier
        rate = np.where(
            notional > 0.0,
            self.cost_per_contract_side / np.where(notional > 0.0, notional, 1.0),
            0.0,
        )
        return traded_arr * rate
