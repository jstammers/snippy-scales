use super::{Fill, FillModel, Order};
use crate::market::MarketEvent;
use crate::portfolio::Portfolio;
use std::collections::VecDeque;
pub struct BacktestEngine<F: FillModel> {
    pub portfolio: Portfolio,
    pub fill_model: F,
    pending_orders: VecDeque<Order>,
}
impl<F: FillModel> BacktestEngine<F> {
    pub fn new(initial_cash: f64, fill_model: F) -> Self {
        Self {
            portfolio: Portfolio::new(initial_cash),
            fill_model,
            pending_orders: VecDeque::new(),
        }
    }
    pub fn submit_order(&mut self, order: Order) {
        self.pending_orders.push_back(order);
    }
    pub fn step(&mut self, event: &MarketEvent) -> Vec<Fill> {
        use crate::market::MarketEventKind::Bar;
        let market_price = match &event.kind {
            Bar(b) => b.close,
        };
        let orders: Vec<Order> = self.pending_orders.drain(..).collect();
        let mut fills = Vec::new();
        for order in orders {
            let mut fill = self.fill_model.simulate(&order, market_price, 0.0);
            fill.ts_filled = event.ts_event;
            self.portfolio.apply_fill(&fill);
            fills.push(fill);
        }
        fills
    }
}
