pub fn ns_to_secs(ns: i64) -> f64 {
    ns as f64 / 1_000_000_000.0
}
pub fn now_ns() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as i64
}
