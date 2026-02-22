use super::Bar;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketEventKind { Bar(Bar) }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEvent { pub instrument_id: u64, pub ts_event: i64, pub kind: MarketEventKind }
