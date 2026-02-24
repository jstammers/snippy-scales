"""Parameter search strategies for hyperparameter optimisation.

Three search types are provided:

* :class:`ParameterGrid` — exhaustive Cartesian product over discrete values.
* :class:`RandomSearch` — uniform random sampling from lists or ranges.
* :class:`OptunaSearch` — Bayesian TPE optimisation (requires ``optuna``).

All three are *iterables* that yield ``dict[str, Any]`` parameter sets, so
they can be passed directly to :class:`~snippy_scales.evaluation.runner.EvaluationRunner`.

Usage::

    grid = ParameterGrid({"fast_period": [10, 20], "slow_period": [40, 60]})
    for params in grid:
        strategy = TrendFollowing(**params)

    random = RandomSearch({"lookback": [63, 126, 252], "vol_target": [0.05, 0.10]},
                          n_iter=20, seed=42)
"""

from __future__ import annotations

import itertools
import random as _random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# ── ParameterGrid ─────────────────────────────────────────────────────────────


class ParameterGrid:
    """Exhaustive Cartesian product over discrete parameter values.

    Args:
        param_dict: Mapping of ``parameter_name → list_of_values``.

    Example::

        grid = ParameterGrid({"fast_period": [10, 20, 40], "slow_period": [40, 60]})
        for params in grid:
            # {"fast_period": 10, "slow_period": 40}, {"fast_period": 10, "slow_period": 60}, …
    """

    def __init__(self, param_dict: dict[str, list[Any]]) -> None:
        if not param_dict:
            raise ValueError("param_dict must contain at least one parameter")
        for k, v in param_dict.items():
            if not isinstance(v, list) or len(v) == 0:
                raise ValueError(f"Values for '{k}' must be a non-empty list")
        self._param_dict = param_dict

    def __iter__(self) -> Iterator[dict[str, Any]]:
        keys = list(self._param_dict.keys())
        for values in itertools.product(*self._param_dict.values()):
            yield dict(zip(keys, values, strict=False))

    def __len__(self) -> int:
        count = 1
        for v in self._param_dict.values():
            count *= len(v)
        return count

    def __repr__(self) -> str:
        return f"ParameterGrid({self._param_dict!r})"


# ── RandomSearch ──────────────────────────────────────────────────────────────


class RandomSearch:
    """Random sampling over parameter distributions.

    Each parameter's distribution is specified as a list of discrete values
    to sample from uniformly.  The same seed always produces the same sequence
    of parameter sets.

    Args:
        param_distributions: Mapping of ``parameter_name → list_of_values``.
            A random value is drawn from the list on each iteration.
        n_iter: Number of parameter sets to sample (default 20).
        seed: Random seed for reproducibility (default 42).

    Example::

        rs = RandomSearch(
            {"lookback": [63, 126, 252], "vol_target": [0.05, 0.10, 0.15]},
            n_iter=10,
            seed=0,
        )
        for params in rs:
            strategy = TimeSeriesMomentum(**params)
    """

    def __init__(
        self,
        param_distributions: dict[str, list[Any]],
        n_iter: int = 20,
        seed: int = 42,
    ) -> None:
        if not param_distributions:
            raise ValueError("param_distributions must contain at least one parameter")
        if n_iter < 1:
            raise ValueError(f"n_iter must be ≥ 1, got {n_iter}")
        self._param_distributions = param_distributions
        self._n_iter = n_iter
        self._seed = seed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rng = _random.Random(self._seed)
        for _ in range(self._n_iter):
            yield {k: rng.choice(v) for k, v in self._param_distributions.items()}

    def __len__(self) -> int:
        return self._n_iter

    def __repr__(self) -> str:
        return (
            f"RandomSearch(n_iter={self._n_iter}, seed={self._seed}, "
            f"params={list(self._param_distributions.keys())})"
        )


# ── OptunaSearch ──────────────────────────────────────────────────────────────


class OptunaSearch:
    """Bayesian hyperparameter optimisation via Optuna (TPE sampler).

    Requires ``optuna`` to be installed::

        uv add optuna  # or: pip install snippy-scales[optuna]

    ``param_space`` maps parameter names to tuples describing the search range:

    * ``(low, high)`` → integer range, sampled with ``trial.suggest_int``
    * ``(low, high, "float")`` → continuous float range, sampled with
      ``trial.suggest_float``
    * ``[v1, v2, …]`` → categorical, sampled with ``trial.suggest_categorical``

    Args:
        param_space: Mapping of parameter name → range spec (see above).
        n_trials: Number of Optuna trials to run (default 50).
        seed: Sampler seed for reproducibility (default 42).
        direction: Optuna optimisation direction — ``"maximize"`` (default,
            maximises mean test Sharpe) or ``"minimize"``.

    Example::

        search = OptunaSearch(
            param_space={
                "fast_period": (5, 50),
                "slow_period": (20, 200),
                "vol_target": (0.05, 0.20, "float"),
            },
            n_trials=100,
        )
        result = runner.evaluate(TrendFollowing, bars, params=search)
    """

    def __init__(
        self,
        param_space: dict[str, tuple[Any, ...] | list[Any]],
        n_trials: int = 50,
        seed: int = 42,
        direction: str = "maximize",
    ) -> None:
        self._check_optuna()
        if not param_space:
            raise ValueError("param_space must contain at least one parameter")
        if n_trials < 1:
            raise ValueError(f"n_trials must be ≥ 1, got {n_trials}")
        self._param_space = param_space
        self._n_trials = n_trials
        self._seed = seed
        self._direction = direction

    @staticmethod
    def _check_optuna() -> None:
        import importlib.util

        if importlib.util.find_spec("optuna") is None:
            raise ImportError(
                "OptunaSearch requires optuna. Install it with:\n"
                "  uv add optuna\n"
                "  # or: pip install 'snippy-scales[optuna]'"
            )

    def suggest(self, trial: Any) -> dict[str, Any]:
        """Suggest a parameter set for a single Optuna trial.

        Args:
            trial: An ``optuna.Trial`` object.

        Returns:
            Parameter dict to pass to the strategy constructor.
        """
        params: dict[str, Any] = {}
        for name, spec in self._param_space.items():
            if isinstance(spec, list):
                params[name] = trial.suggest_categorical(name, spec)
            elif len(spec) == 2:  # noqa: PLR2004
                params[name] = trial.suggest_int(name, int(spec[0]), int(spec[1]))
            elif len(spec) == 3 and spec[2] == "float":  # noqa: PLR2004
                params[name] = trial.suggest_float(name, float(spec[0]), float(spec[1]))
            else:
                raise ValueError(
                    f"Invalid param_space spec for '{name}': {spec!r}.  "
                    "Use (low, high) for int, (low, high, 'float') for float, "
                    "or [v1, v2, …] for categorical."
                )
        return params

    @property
    def n_trials(self) -> int:
        return self._n_trials

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def direction(self) -> str:
        return self._direction

    def __repr__(self) -> str:
        return (
            f"OptunaSearch(n_trials={self._n_trials}, seed={self._seed}, "
            f"params={list(self._param_space.keys())})"
        )
