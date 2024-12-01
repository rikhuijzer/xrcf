//! Structured control flow dialect.
//!
//! Being structured means that the control flow has a structure unlike, for
//! example, `goto`s or `assert`s. Unstructured control flow operations are
//! located in the `cf` (control flow) dialect.

mod op;

use crate::Dialect;

pub use op::IfOp;

pub struct Scf;

impl Dialect for Scf {
    fn name(&self) -> &'static str {
        "scf"
    }
    fn description(&self) -> &'static str {
        "Structured control flow dialect"
    }
}
