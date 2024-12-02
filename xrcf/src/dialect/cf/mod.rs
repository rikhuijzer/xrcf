//! Unstructured control flow dialect.
//!
//! This dialect contains low-level control flow constructs. For example,
//! `cf.br` is a branch operation that unconditionally branches to a given
//! block, like a `goto`.

mod op;

use crate::Dialect;

pub use op::BranchOp;
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
