"""Walk-forward train/test splitting for time-series backtests.

Wraps ``sklearn.model_selection.TimeSeriesSplit`` (expanding window) and adds
a rolling-window variant.  Returns typed :class:`SplitFold` objects that carry
the actual bar DataFrames so callers never need to manage index arithmetic.

Usage::

    splitter = WalkForwardSplit(n_splits=5, test_size=0.2)
    for fold in splitter.split(bars):
        train_result = runner.run(strategy, fold.train)
        test_result  = runner.run(strategy, fold.test)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

if TYPE_CHECKING:
    import polars as pl


# ── SplitFold ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SplitFold:
    """One walk-forward fold containing train and test bar DataFrames.

    Attributes:
        fold_idx: Zero-based fold index.
        train: Training-window bar DataFrame.
        test: Out-of-sample test bar DataFrame.
        train_start: ISO date string for the start of the training window.
        train_end: ISO date string for the end of the training window.
        test_start: ISO date string for the start of the test window.
        test_end: ISO date string for the end of the test window.
    """

    fold_idx: int
    train: pl.DataFrame
    test: pl.DataFrame
    train_start: str
    train_end: str
    test_start: str
    test_end: str


# ── WalkForwardSplit ──────────────────────────────────────────────────────────


class WalkForwardSplit:
    """Walk-forward train/test splitter for time-series bar data.

    Two window strategies are supported:

    * ``"expanding"`` (default): the training window grows with each fold.
      Uses ``sklearn.model_selection.TimeSeriesSplit`` under the hood.
    * ``"rolling"``: the training window has a fixed number of bars.  Each
      fold advances by the same step size as the test window.

    Args:
        n_splits: Number of walk-forward folds (default 5).
        test_size: Fraction of total bars reserved for each test window
            (default 0.2).  Alternatively an integer number of bars.
        gap: Bars to skip between the end of the training window and the
            start of the test window (default 0).  Use this to avoid any
            lookahead from features that require a warmup period.
        window: ``"expanding"`` or ``"rolling"`` (default ``"expanding"``).
        min_train_size: Minimum number of bars required in the training
            window.  Folds with fewer training bars are silently skipped.

    Example::

        splitter = WalkForwardSplit(n_splits=5, window="expanding")
        folds = splitter.split(bars)
        print(f"{len(folds)} folds, first test window: {folds[0].test_start}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float | int = 0.2,
        gap: int = 0,
        window: str = "expanding",
        min_train_size: int | None = None,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be ≥ 1, got {n_splits}")
        if window not in ("expanding", "rolling"):
            raise ValueError(f"window must be 'expanding' or 'rolling', got {window!r}")
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.window = window
        self.min_train_size = min_train_size

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def split(self, bars: pl.DataFrame) -> list[SplitFold]:
        """Split a single-asset bar DataFrame into walk-forward folds.

        Args:
            bars: OHLCV bar DataFrame.  Must contain a timestamp column
                (``ts_event``, ``ts``, or ``timestamp``).

        Returns:
            List of :class:`SplitFold` objects, one per fold.

        Raises:
            ValueError: If the DataFrame has too few rows to form any fold.
        """
        n = len(bars)
        indices = list(range(n))
        folds: list[SplitFold] = []

        for fold_idx, (train_idx, test_idx) in enumerate(self._raw_splits(indices, n)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if self.min_train_size is not None and len(train_idx) < self.min_train_size:
                continue

            train_df = bars[train_idx]
            test_df = bars[test_idx]
            folds.append(
                SplitFold(
                    fold_idx=fold_idx,
                    train=train_df,
                    test=test_df,
                    train_start=self._ts_str(train_df, 0),
                    train_end=self._ts_str(train_df, -1),
                    test_start=self._ts_str(test_df, 0),
                    test_end=self._ts_str(test_df, -1),
                )
            )

        return folds

    def split_multi(
        self, multi_bars: dict[str, pl.DataFrame]
    ) -> list[tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]]:
        """Split a multi-asset bar mapping into walk-forward folds.

        All assets are sliced using the same index positions, preserving
        cross-sectional alignment.  The reference asset for computing split
        positions is the first key in the mapping.

        Args:
            multi_bars: Mapping of ``symbol → bar DataFrame``.  All DataFrames
                must have the same number of rows.

        Returns:
            List of ``(train_multi_bars, test_multi_bars)`` tuples.
        """
        symbols = list(multi_bars.keys())
        if not symbols:
            return []
        ref = multi_bars[symbols[0]]
        n = len(ref)
        indices = list(range(n))
        result: list[tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]] = []

        for _fold_idx, (train_idx, test_idx) in enumerate(self._raw_splits(indices, n)):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if self.min_train_size is not None and len(train_idx) < self.min_train_size:
                continue
            train_bars = {sym: multi_bars[sym][train_idx] for sym in symbols}
            test_bars = {sym: multi_bars[sym][test_idx] for sym in symbols}
            result.append((train_bars, test_bars))

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _raw_splits(self, indices: list[int], n: int) -> list[tuple[list[int], list[int]]]:
        """Return (train_indices, test_indices) pairs for each fold."""
        test_n = self._resolve_test_size(n)

        if self.window == "expanding":
            tscv = TimeSeriesSplit(
                n_splits=self.n_splits,
                test_size=test_n,
                gap=self.gap,
            )
            return [(list(tr), list(te)) for tr, te in tscv.split(np.arange(n))]

        # Rolling window: fixed training size = total - n_splits * test_n - gap
        step = test_n + self.gap
        train_size = n - self.n_splits * step
        if train_size < 1:
            return []

        splits = []
        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + train_size
            test_start = train_end + self.gap
            test_end = test_start + test_n
            if test_end > n:
                break
            splits.append((list(range(train_start, train_end)), list(range(test_start, test_end))))
        return splits

    def _resolve_test_size(self, n: int) -> int:
        """Convert ``test_size`` to an integer number of bars.

        Caps the test window so that the initial training set has at least
        one bar: ``n_splits × test_n < n - gap × n_splits``.
        """
        if isinstance(self.test_size, float):
            requested = max(1, int(n * self.test_size))
        else:
            requested = int(self.test_size)
        # sklearn requires: n - gap - n_splits * test_n > 0
        max_test_n = max(1, (n - self.gap * self.n_splits - 1) // self.n_splits)
        return min(requested, max_test_n)

    @staticmethod
    def _ts_str(df: pl.DataFrame, row: int) -> str:
        """Return a date string for the first/last row of a DataFrame."""
        for col in ("ts_event", "ts", "timestamp", "date"):
            if col in df.columns:
                val = df[col][row]
                # Nanosecond integer timestamp → ISO date
                if isinstance(val, int):
                    import datetime

                    dt = datetime.datetime.fromtimestamp(val / 1e9, tz=datetime.UTC)
                    return dt.strftime("%Y-%m-%d")
                return str(val)
        return ""
