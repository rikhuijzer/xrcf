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
use std::sync::RwLock;

// See `include/mlir/IR/BuiltinOps.h` and goto definition of
// `mlir/IR/BuiltinOps.h.inc`.
pub struct ModuleOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for ModuleOp {
    fn operation_name() -> OperationName {
        OperationName::new("module".to_string())
    }
    fn from_operation(operation: Arc<RwLock<Operation>>) -> Result<Self> {
        if operation.read().unwrap().name() != Self::operation_name().name() {
            return Err(anyhow::anyhow!(
                "Expected module, got {}",
                operation.read().unwrap().name()
            ));
        }
        Ok(Self {
            operation: operation,
        })
    }
    fn set_indentation(&self, indentation: i32) {
        let mut operation = self.operation.write().unwrap();
        operation.set_indentation(indentation);
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
}

impl Display for ModuleOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.operation().read().unwrap())
    }
}

impl ModuleOp {
    pub fn get_body_region(&self) -> Result<Arc<RwLock<Region>>> {
        Ok(self.operation().read().unwrap().region())
    }
    pub fn first_op(&self) -> Result<Arc<dyn Op>> {
        let body_region = self.get_body_region()?;
        let binding = body_region.read().unwrap();
        let blocks = binding.blocks();
        let block = match blocks.first() {
            Some(block) => block,
            None => return Err(anyhow::anyhow!("Expected 1 block in module, got 0")),
        };
        let ops = block.read().unwrap().ops();
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
