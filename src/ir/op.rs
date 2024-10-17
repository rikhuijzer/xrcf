use crate::convert::RewriteResult;
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
    fn verify(&self) -> Result<()> {
        Ok(())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self>
    where
        Self: Sized;
    fn from_operation(operation: Arc<RwLock<Operation>>) -> Result<Self>
    where
        Self: Sized,
    {
        let name = Self::operation_name();
        let op = Self::from_operation_without_verify(operation, name)?;
        op.verify()?;
        Ok(op)
    }
    fn as_any(&self) -> &dyn std::any::Any;
    fn operation(&self) -> &Arc<RwLock<Operation>>;
    /// Returns the name of the operation.
    /// This is a convenience method for `self.operation().name()`.
    /// Unlike `self.operation_name()`, this method is available on a `dyn Op`.
    fn name(&self) -> OperationName {
        let operation = self.operation().read().unwrap();
        operation.name()
    }
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
    fn canonicalize(&self) -> RewriteResult {
        RewriteResult::Unchanged
    }
    fn is_terminator(&self) -> bool {
        false
    }
    fn is_func(&self) -> bool {
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
    /// Replace self with `new` by moving the results of the old operation to
    /// the results of the specified new op, and pointing the `result.defining_op` to the new op.
    /// In effect, this makes all the uses of the old op refer to the new op instead.
    ///
    /// Note that this function assumes that `self` will be dropped after this function call.
    /// Therefore, the old op can still have references to objects that are now part of
    /// the new op.
    fn replace(&self, new: Arc<RwLock<dyn Op>>) {
        let results = {
            let old_operation = self.operation().try_read().unwrap();
            old_operation.results()
        };
        {
            for result in results.try_read().unwrap().iter() {
                let mut result = result.try_write().unwrap();
                result.set_defining_op(Some(new.clone()));
            }
            let new_read = new.try_read().unwrap();
            let mut new_operation = new_read.operation().try_write().unwrap();
            new_operation.set_results(results.clone());
        }
        let old_operation = self.operation().try_read().unwrap();
        // Root ops do not have a parent, so we don't need to update the parent.
        match old_operation.parent() {
            Some(parent) => {
                let parent = parent.try_read().unwrap();
                parent.replace(self.operation().clone(), new.clone());
            }
            None => {}
        }
    }
    /// Return ops that are children of this op (inside blocks that are inside the region).
    fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        let mut result = Vec::new();
        let region = self.region();
        if let Some(region) = region {
            for block in region.read().unwrap().blocks() {
                let block = block.read().unwrap();
                let ops = block.ops();
                let ops = ops.read().unwrap();
                for op in ops.iter() {
                    result.push(op.clone());
                }
            }
        }
        result
    }
    /// Display the operation with the given indentation.
    ///
    /// This method is usually called on a top-level op via `Display::fmt`,
    /// which then calls `display` with `indent` 0.  Next, this method calls
    /// `display` recursively while continuously increasing the indentation
    /// level.
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
