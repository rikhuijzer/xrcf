use crate::ir::Operation;
use anyhow::Result;

/// This is the trait that is implemented by all operations.
/// FuncOp, for example, will be implemented by various dialects.
/// Note that the parser will parse the tokens into an `Operation`
/// and MLIR would cast the `Operation` into a specific `Op` variant
/// such as `FuncOp`.
pub trait Op {
    fn from_operation(operation: Operation) -> Result<Self>
    where
        Self: Sized;
    fn operation(&self) -> Operation;
    fn name(&self) -> &'static str;
}