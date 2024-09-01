pub mod attribute;
pub mod block;
pub mod op;
pub mod operation;
pub mod region;
pub mod value;

pub use crate::dialect::func::FuncOp;
pub use attribute::AnyAttr;
pub use attribute::Attribute;
pub use attribute::StrAttr;
pub use block::Block;
pub use op::Op;
pub use operation::Operation;
pub use operation::OperationName;
pub use region::Region;
pub use value::BlockArgument;
pub use value::OpResult;
pub use value::Type;
pub use value::Value;

use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::pin::Pin;
use std::sync::Arc;

// See `include/mlir/IR/BuiltinOps.h` and goto definition of
// `mlir/IR/BuiltinOps.h.inc`.
pub struct ModuleOp {
    operation: Pin<Box<Operation>>,
}

impl Op for ModuleOp {
    fn operation_name() -> OperationName {
        OperationName::new("module".to_string())
    }
    fn from_operation(operation: Pin<Box<Operation>>) -> Result<Self> {
        if operation.name() != Self::operation_name().name() {
            return Err(anyhow::anyhow!("Expected module, got {}", operation.name()));
        }
        Ok(Self {
            operation: operation,
        })
    }
    fn operation(&self) -> &Pin<Box<Operation>> {
        &self.operation
    }
}

impl Display for ModuleOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.operation())
    }
}

impl ModuleOp {
    pub fn get_body_region(&self) -> Result<&Region> {
        Ok(&self.operation.region())
    }
    pub fn first_op(&self) -> Result<Arc<dyn Op>> {
        let body_region = self.get_body_region()?;
        let block = *body_region.blocks().first().unwrap();
        let ops = block.ops();
        let op = ops.first();
        if op.is_none() {
            return Err(anyhow::anyhow!("Expected 1 op, got 0"));
        }
        Ok(op.unwrap().clone())
    }
}

struct Tmp {
    operation: Pin<Box<Operation>>,
}

impl Tmp {
    fn new(operation: Pin<Box<Operation>>) -> Self {
        Self {
            operation: operation,
        }
    }

    fn modify_operation(&mut self) {
        // Example modification
        let mut op = self.operation.as_mut();
        // Perform modifications on op
        op.rename("new_name".to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_op() {
        let name = crate::ir::operation::OperationName::new("module".to_string());
        let attributes = vec![];
        let mut operation = Operation::default();
        operation.set_name(name).set_attributes(attributes);
        let mut tmp = Tmp::new(Box::pin(operation));
        tmp.modify_operation();
        assert_eq!(tmp.operation.name(), "new_name");
    }
}
