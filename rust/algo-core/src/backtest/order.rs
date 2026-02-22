use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderSide { Buy, Sell }
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderKind { Market, Limit }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order { pub id: u64, pub instrument_id: u64, pub side: OrderSide, pub kind: OrderKind, pub quantity: f64, pub limit_price: Option<f64> }
