use crate::convert::RewriteResult;
use crate::ir::Attribute;
use crate::ir::Block;
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
        operation.wr().set_name(Self::operation_name());
        Self::new(operation)
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
        Self::from_operation_arc(Shared::new(operation.into()))
    }
    fn as_any(&self) -> &dyn std::any::Any;
    fn operation(&self) -> &Arc<RwLock<Operation>>;
    fn name(&self) -> OperationName {
        self.operation().rd().name()
    }
    fn region(&self) -> Option<Arc<RwLock<Region>>> {
        self.operation().rd().region()
    }
    /// Returns the values which this `Op` assigns to.
    /// For most ops, this is the `results` field of type `OpResult`.
    /// But for some other ops like `FuncOp`, this can be different.
    fn assignments(&self) -> Result<Values> {
        Ok(self.operation().rd().results())
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
        Some(self.operation().rd().attributes().get(key)?)
    }
    /// Insert `earlier` before `self` inside `self`'s parent block.
    fn insert_before(&self, earlier: Arc<RwLock<dyn Op>>) {
        let operation = self.operation();
        let parent = match operation.rd().parent() {
            Some(block) => block,
            None => panic!("no parent for {}", self.name()),
        };
        let later = operation.clone();
        parent.rd().insert_before(earlier, later);
    }
    /// Insert `later` after `self` inside `self`'s parent block.
    fn insert_after(&self, later: Arc<RwLock<dyn Op>>) {
        let operation = self.operation();
        let parent = match operation.rd().parent() {
            Some(block) => block,
            None => panic!("no parent for {}", self.name()),
        };
        let earlier = operation.clone();
        parent.rd().insert_after(earlier, later);
    }
    /// Remove the operation from its parent block.
    fn remove(&self) {
        let operation = self.operation();
        let parent = operation.rd().parent().expect("no parent");
        parent.rd().remove(operation.clone());
    }
    /// Replace self with `new` by moving the results of the old operation to
    /// the results of the specified new op, and pointing the
    /// `result.defining_op` to the new op. In effect, this makes all the uses
    /// of the old op refer to the new op instead.
    ///
    /// Note that this function assumes that `self` will be dropped after this
    /// function call. Therefore, the old op can still have references to
    /// objects that are now part of the new op.
    fn replace(&self, new: Arc<RwLock<dyn Op>>) {
        let parent = self.operation().rd().parent();
        let results = self.operation().rd().results();
        for result in results.clone().into_iter() {
            if let Value::OpResult(res) = &mut *result.wr() {
                res.set_defining_op(Some(new.clone()));
            }
        }
        new.rd().operation().wr().set_results(results);
        // Root ops do not have a parent, so in that case we don't need to
        // update the parent.
        if let Some(parent) = parent {
            parent.rd().replace(self.operation().clone(), new.clone())
        }
    }
    /// Return ops that are children of this op (inside blocks that are inside
    /// the region).
    ///
    /// Some ops may decide to override this implementation if the children are
    /// not located inside the main region of the op. For example, the `scf.if`
    /// operation contains two regions: the "then" region and the "else" region.
    fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        if let Some(region) = self.region() {
            region.rd().ops()
        } else {
            vec![]
        }
    }
    fn parent_op(&self) -> Option<Arc<RwLock<dyn Op>>> {
        self.operation().rd().parent_op()
    }
    fn set_parent(&self, parent: Arc<RwLock<Block>>) {
        self.operation().wr().set_parent(Some(parent));
    }
    /// Return the result at the given index.
    ///
    /// Convenience function which makes it easier to set an operand to the
    /// result of an operation.
    fn result(&self, index: usize) -> Arc<RwLock<Value>> {
        self.operation().rd().result(index).unwrap()
    }
    /// Display the operation with the given indentation.
    ///
    /// This method is usually called on a top-level op via `Display::fmt`,
    /// which then calls `display` with `indent` 0.  Next, this method calls
    /// `display` recursively while continuously increasing the indentation
    /// level.
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        self.operation().rd().display(f, 0)
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
        self.0.rd().set_parent(parent);
        self.0.clone()
    }
}
