use crate::convert::RewriteResult;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::GuardedBlock;
use crate::ir::GuardedRegion;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Region;
use crate::ir::Value;
use crate::ir::Values;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// A specific operation.
///
/// See [Operation] for more information about the relationship between
/// [Operation] and [Op].
pub trait Op {
    fn operation_name() -> OperationName
    where
        Self: Sized;
    /// Create an new [Op] from an [Operation].
    ///
    /// This method has to be implemented by all ops and is just used to
    /// create a new [Op]. Do not call this method directly, but rather use
    /// [Self::from_operation].
    fn new(operation: Arc<RwLock<Operation>>) -> Self
    where
        Self: Sized;
    /// Create an [Op] from an [Operation] wrapped in an [Arc].
    ///
    /// See [Self::from_operation] for more information.
    fn from_operation_arc(operation: Arc<RwLock<Operation>>) -> Self
    where
        Self: Sized,
    {
        {
            let name = Self::operation_name();
            let operation_write = operation.clone();
            let mut operation_write = operation_write.wr();
            operation_write.set_name(name);
        }
        let op = Self::new(operation);
        op
    }
    /// Create an [Op] from an [Operation].
    ///
    /// The default implementation for this method automatically sets the name
    /// of the operation to the name of the op. This duplication of the name is
    /// necessary because it allows showing the operation name even when the
    /// [Operation] is not wrapped inside an [Op].
    fn from_operation(operation: Operation) -> Self
    where
        Self: Sized,
    {
        let operation = Shared::new(operation.into());
        Self::from_operation_arc(operation)
    }
    fn as_any(&self) -> &dyn std::any::Any;
    fn operation(&self) -> &Arc<RwLock<Operation>>;
    /// Returns the name of the operation.
    /// This is a convenience method for `self.operation().name()`.
    /// Unlike `self.operation_name()`, this method is available on a `dyn Op`.
    fn name(&self) -> OperationName {
        let operation = self.operation().re();
        operation.name()
    }
    fn region(&self) -> Option<Arc<RwLock<Region>>> {
        let operation = self.operation().re();
        operation.region()
    }
    /// Returns the values which this `Op` assigns to.
    /// For most ops, this is the `results` field of type `OpResult`.
    /// But for some other ops like `FuncOp`, this can be different.
    fn assignments(&self) -> Result<Values> {
        let operation = self.operation();
        let operation = operation.re();
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
    fn is_const(&self) -> bool {
        false
    }
    fn is_pure(&self) -> bool {
        false
    }
    fn attribute(&self, key: &str) -> Option<Arc<dyn Attribute>> {
        let operation = self.operation().re();
        let attributes = operation.attributes();
        let attributes = attributes.map();
        let attributes = attributes.re();
        let value = attributes.get(key)?;
        Some(value.clone())
    }
    /// Insert `earlier` before `self` inside `self`'s parent block.
    fn insert_before(&self, earlier: Arc<RwLock<dyn Op>>) {
        let operation = self.operation().re();
        let block = operation.parent().expect("no parent");
        let block = block.re();
        let later = self.operation().clone();
        block.insert_before(earlier, later);
    }
    /// Insert `later` after `self` inside `self`'s parent block.
    fn insert_after(&self, later: Arc<RwLock<dyn Op>>) {
        let operation = self.operation().re();
        let block = match operation.parent() {
            Some(block) => block,
            None => panic!("no parent for {}", self.name()),
        };
        let earlier = self.operation().clone();
        block.insert_after(earlier, later);
    }
    /// Remove the operation from its parent block.
    fn remove(&self) {
        let operation = self.operation().re();
        let block = operation.parent().expect("no parent");
        let block = block.re();
        block.remove(self.operation().clone());
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
            let old_operation = self.operation().re();
            old_operation.results()
        };
        {
            for result in results.vec().re().iter() {
                let mut result = result.wr();
                if let Value::OpResult(res) = &mut *result {
                    res.set_defining_op(Some(new.clone()));
                }
            }
            let new_read = new.re();
            let mut new_operation = new_read.operation().wr();
            new_operation.set_results(results.clone());
        }
        let old_operation = self.operation().re();
        // Root ops do not have a parent, so we don't need to update the parent.
        match old_operation.parent() {
            Some(parent) => {
                let parent = parent.re();
                parent.replace(self.operation().clone(), new.clone());
            }
            None => {}
        }
    }
    /// Return ops that are children of this op (inside blocks that are inside the region).
    ///
    /// Some ops may decide to override this implementation if the children are
    /// not located inside the main region of the op. For example, the `scf.if`
    /// operation contains two regions: the "then" region and the "else" region.
    fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        let region = self.region();
        if let Some(region) = region {
            region.ops()
        } else {
            vec![]
        }
    }
    fn parent_op(&self) -> Option<Arc<RwLock<dyn Op>>> {
        let operation = self.operation().re();
        operation.parent_op()
    }
    fn set_parent(&self, parent: Arc<RwLock<Block>>) {
        let operation = self.operation();
        let mut operation = operation.wr();
        operation.set_parent(Some(parent));
    }
    /// Return the result at the given index.
    ///
    /// Convenience function which makes it easier to set an operand to the
    /// result of an operation.
    fn result(&self, index: usize) -> Arc<RwLock<Value>> {
        let operation = self.operation().re();
        let value = operation.result(index).unwrap();
        value
    }
    /// Display the operation with the given indentation.
    ///
    /// This method is usually called on a top-level op via `Display::fmt`,
    /// which then calls `display` with `indent` 0.  Next, this method calls
    /// `display` recursively while continuously increasing the indentation
    /// level.
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().re();
        operation.display(f, 0)
    }
}

impl Display for dyn Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

impl<T: ParserDispatch> Parser<T> {
    pub fn parse_op(&mut self, parent: Option<Arc<RwLock<Block>>>) -> Result<Arc<RwLock<dyn Op>>> {
        T::parse_op(self, parent)
    }
}

#[must_use = "the object inside `UnsetOp` should be further initialized, see the setter methods"]
pub struct UnsetOp(Arc<RwLock<dyn Op>>);

impl UnsetOp {
    pub fn new(op: Arc<RwLock<dyn Op>>) -> Self {
        UnsetOp(op)
    }
    pub fn set_parent(&self, parent: Arc<RwLock<Block>>) -> Arc<RwLock<dyn Op>> {
        let op = self.0.re();
        op.set_parent(parent);
        self.0.clone()
    }
}

pub trait GuardedOp {
    fn insert_after(&self, later: Arc<RwLock<dyn Op>>);
    fn insert_before(&self, earlier: Arc<RwLock<dyn Op>>);
    fn is_const(&self) -> bool;
    fn name(&self) -> OperationName;
    fn operation(&self) -> Arc<RwLock<Operation>>;
    fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>>;
    fn parent_op(&self) -> Option<Arc<RwLock<dyn Op>>>;
    fn remove(&self);
    fn replace(&self, new: Arc<RwLock<dyn Op>>);
    fn result(&self, index: usize) -> Arc<RwLock<Value>>;
    fn set_parent(&self, parent: Arc<RwLock<Block>>);
}

impl GuardedOp for Arc<RwLock<dyn Op>> {
    fn insert_after(&self, later: Arc<RwLock<dyn Op>>) {
        self.re().insert_after(later);
    }
    fn insert_before(&self, earlier: Arc<RwLock<dyn Op>>) {
        self.re().insert_before(earlier);
    }
    fn is_const(&self) -> bool {
        self.re().is_const()
    }
    fn name(&self) -> OperationName {
        self.re().name()
    }
    fn operation(&self) -> Arc<RwLock<Operation>> {
        self.re().operation().clone()
    }
    fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        self.re().ops()
    }
    fn parent_op(&self) -> Option<Arc<RwLock<dyn Op>>> {
        self.re().parent_op()
    }
    fn remove(&self) {
        self.re().remove();
    }
    fn replace(&self, new: Arc<RwLock<dyn Op>>) {
        self.re().replace(new);
    }
    fn result(&self, index: usize) -> Arc<RwLock<Value>> {
        self.re().result(index)
    }
    fn set_parent(&self, parent: Arc<RwLock<Block>>) {
        self.re().set_parent(parent);
    }
}
