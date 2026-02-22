use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar { pub ts_event: i64, pub open: f64, pub high: f64, pub low: f64, pub close: f64, pub volume: f64 }
