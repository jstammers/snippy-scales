"""Classical SDE calibration and simulation.

Baselines that any neural SDE has to beat.  They are cheap to fit, have few
parameters, and — critically — their parameters can be checked for *stability*
across walk-forward folds.  A neural model fitted to data whose classical
parameters wander from fold to fold is fitting noise, and no amount of
architecture will fix that.

Models:

* **Ornstein–Uhlenbeck** — mean-reverting Gaussian process.  Fitted here by
  *exact* maximum likelihood using the closed-form transition density, not by
  the usual Euler/OLS approximation, so estimates stay unbiased at coarse
  sampling intervals.  The natural model for a spread or for log-volatility.
* **Heston** — stochastic volatility with CIR variance.  The standard
  benchmark for "volatility is a Markov diffusion".
* **Rough Bergomi** — fractional stochastic volatility with Hurst ``H``.
  The rough-volatility alternative; reduces to a Bergomi-type model at
  ``H = 0.5``.

Simulators are provided so that a model can be tested against synthetic data
whose ground truth is known — the only way to distinguish an estimator bug from
a genuine market finding.  ``numpy``/``scipy`` only; no deep-learning
dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "OUParams",
    "fit_ou",
    "simulate_ou",
    "HestonParams",
    "simulate_heston",
    "RoughBergomiParams",
    "simulate_rough_bergomi",
    "fractional_gaussian_noise",
]


# ── Ornstein–Uhlenbeck ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OUParams:
    """Parameters of ``dX = kappa (theta - X) dt + sigma dW``.

    Attributes:
        kappa: Mean-reversion speed (per unit time).  Larger means faster.
        theta: Long-run mean level.
        sigma: Instantaneous volatility.
    """

    kappa: float
    theta: float
    sigma: float

    @property
    def half_life(self) -> float:
        """Time for a deviation to decay by half: ``ln 2 / kappa``.

        Returns:
            Half-life in the same time units as ``kappa``; ``inf`` when
            ``kappa`` is non-positive (no mean reversion).
        """
        if self.kappa <= 0.0:
            return math.inf
        return math.log(2.0) / self.kappa

    @property
    def stationary_std(self) -> float:
        """Standard deviation of the stationary distribution.

        Returns:
            ``sigma / sqrt(2 kappa)``; ``inf`` without mean reversion.
        """
        if self.kappa <= 0.0:
            return math.inf
        return self.sigma / math.sqrt(2.0 * self.kappa)


def fit_ou(series: np.ndarray, dt: float = 1.0) -> OUParams:
    """Fit an OU process by exact maximum likelihood.

    The OU transition density is Gaussian in closed form:

    .. math::

        X_{t+\\Delta} \\mid X_t \\sim N\\!\\left(
            \\theta + (X_t - \\theta)e^{-\\kappa\\Delta},\\;
            \\frac{\\sigma^2}{2\\kappa}\\left(1 - e^{-2\\kappa\\Delta}\\right)
        \\right)

    so the MLE has an analytic solution via the lag-1 regression
    :math:`X_{t+1} = a + b X_t + \\varepsilon` with
    :math:`\\kappa = -\\ln b / \\Delta`.  Using the exact density rather than
    an Euler discretisation matters at daily sampling, where the Euler
    approximation biases ``kappa`` noticeably upward.

    Args:
        series: Observed path, equally spaced.
        dt: Time between observations, in the units ``kappa`` is expressed in.
            Use ``1/252`` for daily data to get an annualised ``kappa``.

    Returns:
        Fitted :class:`OUParams`.  When the lag-1 slope is non-positive the
        series shows no mean reversion at this sampling interval; ``kappa`` is
        returned as ``0.0`` and ``sigma`` from the raw increments.

    Raises:
        ValueError: If *series* has fewer than 3 points or *dt* is not positive.
    """
    path = np.asarray(series, dtype=np.float64)
    if path.size < 3:
        raise ValueError(f"series must have at least 3 points, got {path.size}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    x_prev = path[:-1]
    x_next = path[1:]
    n = x_prev.size

    mean_prev = float(np.mean(x_prev))
    mean_next = float(np.mean(x_next))
    cov = float(np.mean((x_prev - mean_prev) * (x_next - mean_next)))
    var_prev = float(np.mean((x_prev - mean_prev) ** 2))

    if var_prev <= 0.0:
        return OUParams(kappa=0.0, theta=mean_prev, sigma=0.0)

    slope = cov / var_prev
    if slope <= 0.0 or slope >= 1.0:
        # No detectable mean reversion: fall back to a driftless estimate.
        sigma = float(np.std(np.diff(path), ddof=1)) / math.sqrt(dt)
        return OUParams(kappa=0.0, theta=mean_prev, sigma=sigma)

    kappa = -math.log(slope) / dt
    theta = (mean_next - slope * mean_prev) / (1.0 - slope)

    residuals = x_next - (theta + (x_prev - theta) * slope)
    residual_var = float(np.sum(residuals**2) / n)
    denominator = 1.0 - slope**2
    sigma = math.sqrt(max(residual_var * 2.0 * kappa / denominator, 0.0))

    return OUParams(kappa=kappa, theta=theta, sigma=sigma)


def simulate_ou(
    params: OUParams,
    n_steps: int,
    dt: float = 1.0,
    x0: float | None = None,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate OU paths using the exact transition density.

    Exact simulation means the result is distributionally correct at any *dt*,
    so it is a valid ground truth for testing :func:`fit_ou`.

    Args:
        params: Process parameters.
        n_steps: Number of steps to simulate (output has ``n_steps + 1``
            points including the initial value).
        dt: Time step.
        x0: Initial value.  Defaults to ``params.theta``.
        n_paths: Number of independent paths.
        rng: Random generator; a fresh default is used when ``None``.

    Returns:
        Array of shape ``(n_paths, n_steps + 1)``.

    Raises:
        ValueError: If *n_steps* or *n_paths* is not positive, or *dt* is not
            positive.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    generator = rng if rng is not None else np.random.default_rng()
    start = params.theta if x0 is None else x0

    if params.kappa > 0.0:
        decay = math.exp(-params.kappa * dt)
        step_std = params.sigma * math.sqrt((1.0 - decay**2) / (2.0 * params.kappa))
    else:
        decay = 1.0
        step_std = params.sigma * math.sqrt(dt)

    out = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    out[:, 0] = start
    shocks = generator.normal(0.0, step_std, size=(n_paths, n_steps))
    for i in range(n_steps):
        out[:, i + 1] = params.theta + (out[:, i] - params.theta) * decay + shocks[:, i]
    return out


# ── Heston ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HestonParams:
    """Parameters of the Heston stochastic-volatility model.

    .. math::

        dS = \\mu S \\, dt + \\sqrt{V} S \\, dW^S \\\\
        dV = \\kappa(\\theta - V) dt + \\xi \\sqrt{V} \\, dW^V

    with :math:`d\\langle W^S, W^V \\rangle = \\rho \\, dt`.

    Attributes:
        kappa: Variance mean-reversion speed.
        theta: Long-run variance level.
        xi: Volatility of volatility.
        rho: Correlation between price and variance shocks.  Strongly negative
            for equity indices — the leverage effect.
        v0: Initial variance.
        mu: Drift of the price process.
    """

    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    mu: float = 0.0

    @property
    def feller_satisfied(self) -> bool:
        """Whether ``2 kappa theta > xi^2``.

        When the Feller condition holds the variance process stays strictly
        positive.  When it fails the variance touches zero, and a simulation
        scheme that does not truncate will produce NaNs.

        Returns:
            ``True`` if the condition is satisfied.
        """
        return 2.0 * self.kappa * self.theta > self.xi**2


def simulate_heston(
    params: HestonParams,
    n_steps: int,
    dt: float,
    s0: float = 100.0,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Heston price and variance paths.

    Uses a full-truncation Euler scheme, which is the standard robust choice:
    the variance is floored at zero in both the drift and the diffusion, so
    paths remain well-defined even when the Feller condition fails (as it
    routinely does for calibrated equity parameters).

    Args:
        params: Model parameters.
        n_steps: Number of steps.
        dt: Time step (e.g. ``1/252`` for daily).
        s0: Initial price.
        n_paths: Number of independent paths.
        rng: Random generator; a fresh default is used when ``None``.

    Returns:
        Tuple ``(prices, variances)``, each of shape ``(n_paths, n_steps + 1)``.

    Raises:
        ValueError: If *n_steps*, *n_paths* or *dt* is not positive.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    generator = rng if rng is not None else np.random.default_rng()
    sqrt_dt = math.sqrt(dt)

    prices = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    variances = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    prices[:, 0] = s0
    variances[:, 0] = params.v0

    z_v = generator.normal(size=(n_paths, n_steps))
    z_perp = generator.normal(size=(n_paths, n_steps))
    z_s = params.rho * z_v + math.sqrt(max(1.0 - params.rho**2, 0.0)) * z_perp

    for i in range(n_steps):
        v = np.maximum(variances[:, i], 0.0)
        sqrt_v = np.sqrt(v)

        variances[:, i + 1] = (
            variances[:, i]
            + params.kappa * (params.theta - v) * dt
            + params.xi * sqrt_v * sqrt_dt * z_v[:, i]
        )
        prices[:, i + 1] = prices[:, i] * np.exp(
            (params.mu - 0.5 * v) * dt + sqrt_v * sqrt_dt * z_s[:, i]
        )

    return prices, np.maximum(variances, 0.0)


# ── Rough Bergomi ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoughBergomiParams:
    """Parameters of the rough Bergomi model.

    .. math::

        V_t = \\xi_0 \\exp\\!\\left(\\eta W^H_t
              - \\tfrac{1}{2}\\eta^2 t^{2H}\\right)

    where :math:`W^H` is fractional Brownian motion with Hurst index ``H``.

    Attributes:
        hurst: Hurst exponent.  Empirically ~0.1 for equity index volatility;
            ``0.5`` recovers a standard (non-rough) model.
        eta: Volatility of volatility.
        rho: Correlation between the price and volatility drivers.
        xi0: Initial forward variance level.
    """

    hurst: float
    eta: float
    rho: float
    xi0: float


def fractional_gaussian_noise(
    n: int,
    hurst: float,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate fractional Gaussian noise via the Davies–Harte method.

    Exact (not approximate) synthesis using the circulant embedding of the
    autocovariance, so the resulting increments have precisely the intended
    long-range dependence.  That exactness is what makes this usable as ground
    truth for validating :func:`~snippy_scales.research.roughness.hurst_exponent`.

    Falls back to circulant embedding with a non-negativity fix if the exact
    embedding is not positive definite, which can occur for extreme ``H``.

    Args:
        n: Number of increments.
        hurst: Hurst exponent in ``(0, 1)``.
        n_paths: Number of independent paths.
        rng: Random generator; a fresh default is used when ``None``.

    Returns:
        Array of shape ``(n_paths, n)`` of unit-variance fGn increments.

    Raises:
        ValueError: If *hurst* is not in ``(0, 1)`` or *n* is not positive.
    """
    if not 0.0 < hurst < 1.0:
        raise ValueError(f"hurst must be in (0, 1), got {hurst}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    generator = rng if rng is not None else np.random.default_rng()

    # Autocovariance of fGn at lag k.
    k = np.arange(n, dtype=np.float64)
    gamma = 0.5 * (
        np.abs(k - 1.0) ** (2.0 * hurst)
        - 2.0 * np.abs(k) ** (2.0 * hurst)
        + np.abs(k + 1.0) ** (2.0 * hurst)
    )

    # Circulant embedding: [g_0, g_1, ..., g_{n-1}, g_{n-2}, ..., g_1].
    circulant = np.concatenate([gamma, gamma[-2:0:-1]])
    eigenvalues = np.fft.fft(circulant).real
    eigenvalues = np.maximum(eigenvalues, 0.0)  # guard against tiny negatives
    m = eigenvalues.size

    noise = generator.normal(size=(n_paths, m)) + 1j * generator.normal(size=(n_paths, m))
    spectrum = np.sqrt(eigenvalues / (2.0 * m)) * noise
    return np.fft.fft(spectrum, axis=1).real[:, :n] * math.sqrt(2.0)


def simulate_rough_bergomi(
    params: RoughBergomiParams,
    n_steps: int,
    dt: float,
    s0: float = 100.0,
    n_paths: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate rough Bergomi price and variance paths.

    Builds the fractional driver by Riemann–Liouville convolution of fGn, then
    forms the log-normal variance process and the correlated price path.

    Args:
        params: Model parameters.
        n_steps: Number of steps.
        dt: Time step.
        s0: Initial price.
        n_paths: Number of independent paths.
        rng: Random generator; a fresh default is used when ``None``.

    Returns:
        Tuple ``(prices, variances)``, each of shape ``(n_paths, n_steps + 1)``.

    Raises:
        ValueError: If *n_steps*, *n_paths* or *dt* is not positive.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    generator = rng if rng is not None else np.random.default_rng()
    h = params.hurst

    fgn = fractional_gaussian_noise(n_steps, h, n_paths=n_paths, rng=generator)
    # Scale increments to the fBm on this time grid, then accumulate.
    fbm = np.cumsum(fgn, axis=1) * (dt**h)

    times = np.arange(1, n_steps + 1, dtype=np.float64) * dt
    log_variance = params.eta * fbm - 0.5 * params.eta**2 * times ** (2.0 * h)

    variances = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    variances[:, 0] = params.xi0
    variances[:, 1:] = params.xi0 * np.exp(log_variance)

    # Price driver correlated with the volatility driver.
    z_perp = generator.normal(size=(n_paths, n_steps))
    z_price = params.rho * fgn + math.sqrt(max(1.0 - params.rho**2, 0.0)) * z_perp

    prices = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    prices[:, 0] = s0
    sqrt_dt = math.sqrt(dt)
    for i in range(n_steps):
        v = np.maximum(variances[:, i], 0.0)
        prices[:, i + 1] = prices[:, i] * np.exp(
            -0.5 * v * dt + np.sqrt(v) * sqrt_dt * z_price[:, i]
        )

    return prices, variances
