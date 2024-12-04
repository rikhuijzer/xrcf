//! Function dialect.
//!
//! This dialect is meant to hold operations that are related to functions.
mod op;

pub use op::Call;
pub use op::CallOp;
pub use op::Func;
pub use op::FuncOp;
pub use op::ReturnOp;
