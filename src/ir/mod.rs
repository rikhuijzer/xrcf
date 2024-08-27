pub mod block;
pub mod operation;
pub mod region;

pub use operation::Op;
pub use operation::Operation;
pub use region::Region;
pub use block::Block;

use anyhow::Result;

// See `include/mlir/IR/BuiltinOps.h` and goto definition of 
// `mlir/IR/BuiltinOps.h.inc`.
pub struct ModuleOp {
    operation: Operation,
}

impl Op for ModuleOp {
    fn from_operation(operation: Operation) -> Result<Self> {
        if operation.name() != "module" {
            return Err(anyhow::anyhow!("Expected module, got {}", operation.name()));
        }
        if operation.regions().len() != 1 {
            return Err(anyhow::anyhow!("Expected 1 region, got {}",
            operation.regions().len()));
        }
        Ok(Self { operation })
    }
    fn name(&self) -> &'static str {
        "module"
    }
}

impl ModuleOp {
    // somehow get the body (first region?)
    // fn body()
    pub fn getBodyRegion(&self) -> Region {
        self.operation.regions().first().unwrap().clone()
    }
}