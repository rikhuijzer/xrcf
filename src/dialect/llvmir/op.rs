use crate::ir::Operation;
use std::pin::Pin;
use crate::ir::Op;
use anyhow::Result;

struct GlobalOp {
    operation: Pin<Box<Operation>>,
}

impl Op for GlobalOp {
    fn name() -> &'static str {
        "llvm.mlir.global"
    }
    fn from_operation(operation: Pin<Box<Operation>>) -> Result<Self> {
        if operation.name() != Self::name() {
            return Err(anyhow::anyhow!("Expected global, got {}", operation.name()));
        }
        Ok(Self { operation: operation })
    }
    fn operation(&self) -> Pin<Box<Operation>> {
        self.operation.clone()
    }
}