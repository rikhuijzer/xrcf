mod op;

use crate::Dialect;

pub use op::CondBranchOp;

pub struct Cf;

impl Dialect for Cf {
    fn name(&self) -> &'static str {
        "cf"
    }
    fn description(&self) -> &'static str {
        "Low-level control flow"
    }
}
