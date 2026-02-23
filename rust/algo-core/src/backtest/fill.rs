use super::order::Order;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    pub order_id: u64,
    pub instrument_id: u64,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
    pub ts_filled: i64,
}
pub trait FillModel: Send + Sync {
    fn simulate(&self, order: &Order, market_price: f64, volatility: f64) -> Fill;
}
pub struct FlatFillModel {
    pub commission_per_unit: f64,
}
impl FillModel for FlatFillModel {
    fn simulate(&self, order: &Order, market_price: f64, _volatility: f64) -> Fill {
        Fill {
            order_id: order.id,
            instrument_id: order.instrument_id,
            quantity: order.quantity,
            price: market_price,
            commission: order.quantity.abs() * self.commission_per_unit,
            ts_filled: 0,
        }
    }
}
