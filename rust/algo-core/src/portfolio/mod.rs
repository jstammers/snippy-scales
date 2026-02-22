use crate::backtest::Fill;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position { pub instrument_id: u64, pub quantity: f64, pub avg_entry_price: f64 }
#[derive(Debug, Serialize, Deserialize)]
pub struct Portfolio { pub cash: f64, pub positions: HashMap<u64, Position>, pub total_commission: f64 }
impl Portfolio {
    pub fn new(initial_cash: f64) -> Self { Self { cash: initial_cash, positions: HashMap::new(), total_commission: 0.0 } }
    pub fn apply_fill(&mut self, fill: &Fill) {
        self.cash -= fill.price * fill.quantity + fill.commission;
        self.total_commission += fill.commission;
        let pos = self.positions.entry(fill.instrument_id).or_insert(Position { instrument_id: fill.instrument_id, quantity: 0.0, avg_entry_price: fill.price });
        let new_qty = pos.quantity + fill.quantity;
        if new_qty.abs() < 1e-10 { self.positions.remove(&fill.instrument_id); }
        else { pos.avg_entry_price = (pos.avg_entry_price * pos.quantity + fill.price * fill.quantity) / new_qty; pos.quantity = new_qty; }
    }
    pub fn net_liquidation_value(&self, prices: &HashMap<u64, f64>) -> f64 {
        self.cash + self.positions.values().map(|p| p.quantity * prices.get(&p.instrument_id).copied().unwrap_or(p.avg_entry_price)).sum::<f64>()
    }
}
