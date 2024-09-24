use crate::ir::Operation;
use crate::ir::OperationName;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// This is the trait that is implemented by all operations.
/// FuncOp, for example, will be implemented by various dialects.
/// Note that the parser will parse the tokens into an `Operation`
/// and MLIR would cast the `Operation` into a specific `Op` variant
/// such as `FuncOp`.
pub trait Op {
    fn operation_name() -> OperationName
    where
        Self: Sized;
    fn from_operation(operation: Arc<RwLock<Operation>>) -> Result<Self>
    where
        Self: Sized;
    fn set_indent(&self, indent: i32);
    fn operation(&self) -> &Arc<RwLock<Operation>>;
    fn is_terminator(&self) -> bool {
        false
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let operation = self.operation().read().unwrap();
        println!("indent for op: {} {}", indent, operation);
        let indentation = " ".repeat(indent as usize);
        write!(f, "{indentation}")?;
        operation.display(f, indent)
    }
}

impl Display for dyn Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
