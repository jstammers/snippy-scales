use pyo3::prelude::*;
use algo_core::utils::stats;

#[pyfunction]
fn sharpe_ratio(returns: Vec<f64>, risk_free: f64, periods_per_year: f64) -> f64 {
    stats::sharpe_ratio(&returns, risk_free, periods_per_year)
}

#[pyfunction]
fn max_drawdown(equity: Vec<f64>) -> f64 {
    stats::max_drawdown(&equity)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown, m)?)?;
    Ok(())
}
