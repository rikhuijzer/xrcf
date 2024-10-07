use crate::canonicalize::CanonicalizeResult;
use crate::ir::Attribute;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Region;
use crate::ir::Values;
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
    fn as_any(&self) -> &dyn std::any::Any;
    fn operation(&self) -> &Arc<RwLock<Operation>>;
    fn region(&self) -> Option<Arc<RwLock<Region>>> {
        let operation = self.operation().read().unwrap();
        operation.region()
    }
    /// Returns the values which this `Op` assigns to.
    /// For most ops, this is the `results` field of type `OpResult`.
    /// But for some other ops like `FuncOp`, this can be different.
    fn assignments(&self) -> Result<Values> {
        let operation = self.operation();
        let operation = operation.read().unwrap();
        let results = operation.results();
        Ok(results.clone())
    }
    fn canonicalize(&self) -> CanonicalizeResult {
        CanonicalizeResult::Unchanged
    }
    fn is_terminator(&self) -> bool {
        false
    }
    fn attribute(&self, key: &str) -> Option<Arc<dyn Attribute>> {
        let operation = self.operation().read().unwrap();
        let attributes = operation.attributes();
        let attributes = attributes.map();
        let attributes = attributes.read().unwrap();
        let value = attributes.get(key)?;
        Some(value.clone())
    }
    fn insert_before(&self, earlier: Arc<RwLock<dyn Op>>) {
        let operation = self.operation().read().unwrap();
        let block = operation.parent().unwrap();
        let block = block.read().unwrap();
        let later = self.operation().clone();
        block.insert_before(earlier, later);
    }
    /// Replace the results of the old operation with the results of the specified new op.
    fn replace(&self, _new: Arc<RwLock<dyn Op>>) {
        todo!()
        // TODO: Move the results of the new op to the old op.
    }
    fn verify(&self) -> Result<()> {
        Ok(())
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let operation = self.operation().read().unwrap();
        let spaces = crate::ir::spaces(indent);
        write!(f, "{spaces}")?;
        operation.display(f, indent)
    }
}

impl Display for dyn Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
