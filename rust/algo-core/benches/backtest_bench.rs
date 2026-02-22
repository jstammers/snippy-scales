use criterion::{black_box, criterion_group, criterion_main, Criterion};
use algo_core::utils::stats::sharpe_ratio;
fn bench_sharpe(c: &mut Criterion) {
    let returns: Vec<f64> = (0..10_000).map(|i| (i as f64).sin() * 0.01).collect();
    c.bench_function("sharpe_10k", |b| b.iter(|| sharpe_ratio(black_box(&returns), 0.0, 252.0)));
}
criterion_group!(benches, bench_sharpe);
criterion_main!(benches);
