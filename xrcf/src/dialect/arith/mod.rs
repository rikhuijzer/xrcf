//! Arithmetic dialect.
//!
//! This dialect is meant to hold basic integer and floating point operations.
mod op;

use crate::Dialect;

pub use op::AddiOp;
pub use op::ConstantOp;
pub use op::DivsiOp;
pub use op::SubiOp;

pub struct Arith;

impl Dialect for Arith {
    fn name(&self) -> &'static str {
        "arith"
    }
    fn description(&self) -> &'static str {
        "Arithmetic dialect"
    }
}
