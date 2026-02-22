//! Minimal PyO3 wrapper for the backtest engine.
//! Extend as the Rust API stabilises.

use pyo3::prelude::*;
use algo_core::utils::stats;

/// Exposed as a convenience: run a simple vectorised backtest
/// given a list of closes and a list of position sizes (±1, 0).
/// Returns (equity_curve, sharpe, max_drawdown).
#[pyfunction]
fn run_vectorised(
    closes: Vec<f64>,
    positions: Vec<f64>,
    initial_cash: f64,
    commission_per_trade: f64,
) -> PyResult<(Vec<f64>, f64, f64)> {
    if closes.len() != positions.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "closes and positions must be the same length",
        ));
    }
    let mut cash = initial_cash;
    let mut pos = 0.0_f64;
    let mut equity_curve = Vec::with_capacity(closes.len());
    let mut returns = Vec::with_capacity(closes.len());

    for (i, (&close, &target_pos)) in closes.iter().zip(positions.iter()).enumerate() {
        let trade_qty = target_pos - pos;
        if trade_qty.abs() > 1e-10 {
            cash -= trade_qty * close + trade_qty.abs() * commission_per_trade;
        }
        pos = target_pos;
        let equity = cash + pos * close;
        equity_curve.push(equity);
        if i > 0 {
            let prev = equity_curve[i - 1];
            returns.push((equity - prev) / prev.max(1e-10));
        }
    }

    let sharpe = stats::sharpe_ratio(&returns, 0.0, 252.0);
    let mdd = stats::max_drawdown(&equity_curve);
    Ok((equity_curve, sharpe, mdd))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_vectorised, m)?)?;
    Ok(())
}
