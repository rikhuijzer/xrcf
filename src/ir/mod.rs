pub mod block;
pub mod operation;
pub mod region;
pub mod op;

pub use op::Op;
pub use operation::Operation;
pub use region::Region;
pub use block::Block;

use std::pin::Pin;
use anyhow::Result;

// See `include/mlir/IR/BuiltinOps.h` and goto definition of 
// `mlir/IR/BuiltinOps.h.inc`.
pub struct ModuleOp {
    operation: Pin<Box<Operation>>,
}

impl Op for ModuleOp {
    fn name() -> &'static str {
        "module"
    }
    fn from_operation(operation: Pin<Box<Operation>>) -> Result<Self> {
        if operation.name() != Self::name() {
            return Err(anyhow::anyhow!("Expected module, got {}", operation.name()));
        }
        if operation.regions().len() != 1 {
            return Err(anyhow::anyhow!("Expected 1 region, got {}",
            operation.regions().len()));
        }
        Ok(Self { operation: operation })
    }
    fn operation(&self) -> Pin<Box<Operation>> {
        self.operation.clone()
    }
}

impl ModuleOp {
    // somehow get the body (first region?)
    // fn body()
    pub fn get_body_region(&self) -> Region {
        self.operation.regions().first().unwrap().clone()
    }
}

struct Tmp {
    operation: Pin<Box<Operation>>,
}

impl Tmp {
    fn new(operation: Pin<Box<Operation>>) -> Self {
        Self { operation: operation }
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
        let operation = Operation::new(name, vec![], None);
        let mut tmp = Tmp::new(Box::pin(operation));
        tmp.modify_operation();
        assert_eq!(tmp.operation.name(), "new_name");
    }
}