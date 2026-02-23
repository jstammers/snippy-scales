pub fn sharpe_ratio(returns: &[f64], risk_free: f64, periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let std_dev = (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n).sqrt();
    if std_dev < 1e-12 {
        return 0.0;
    }
    (mean - risk_free / periods_per_year) / std_dev * periods_per_year.sqrt()
}
pub fn max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &e in equity {
        if e > peak {
            peak = e;
        }
        let dd = (peak - e) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_flat_returns() {
        let returns = vec![0.01; 252];
        let s = sharpe_ratio(&returns, 0.0, 252.0);
        // Flat returns have zero volatility, so Sharpe should be 0.0
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_sharpe_with_variance() {
        let returns = vec![0.01, 0.02, 0.01, 0.03]; // varying returns
        let s = sharpe_ratio(&returns, 0.0, 252.0);
        assert!(s.is_finite() && s > 0.0);
    }

    #[test]
    fn test_max_drawdown_monotone_up() {
        let equity: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert!((max_drawdown(&equity) - 0.0).abs() < 1e-9);
    }
}
