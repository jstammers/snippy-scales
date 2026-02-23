use crate::backtest::Order;
use crate::portfolio::Portfolio;
#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub max_position_notional: f64,
    pub max_order_notional: f64,
}
pub struct RiskCheck {
    pub limits: RiskLimits,
}
#[derive(Debug)]
pub enum RiskViolation {
    OrderTooLarge,
    PositionLimitBreached,
}
impl RiskCheck {
    pub fn check(
        &self,
        order: &Order,
        price: f64,
        _portfolio: &Portfolio,
    ) -> Result<(), RiskViolation> {
        if order.quantity.abs() * price > self.limits.max_order_notional {
            return Err(RiskViolation::OrderTooLarge);
        }
        Ok(())
    }
}
