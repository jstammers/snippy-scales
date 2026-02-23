use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum InstrumentKind {
    Future,
    Option,
    Spot,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instrument {
    pub symbol: String,
    pub kind: InstrumentKind,
    pub multiplier: f64,
    pub tick_size: f64,
}
