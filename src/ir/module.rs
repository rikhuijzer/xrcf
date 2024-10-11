use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Region;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
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
    fn from_operation_without_verify(operation: Arc<RwLock<Operation>>) -> Result<Self> {
        if operation.read().unwrap().name() != Self::operation_name() {
            return Err(anyhow::anyhow!(
                "Expected module, got {}",
                operation.read().unwrap().name()
            ));
        }
        Ok(Self {
            operation: operation,
        })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let operation = self.operation().read().unwrap();
        let spaces = crate::ir::spaces(indent);
        write!(f, "{spaces}")?;
        operation.display(f, indent + 1)
    }
}

impl Display for ModuleOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.operation().read().unwrap())
    }
}

impl ModuleOp {
    pub fn get_body_region(&self) -> Result<Option<Arc<RwLock<Region>>>> {
        Ok(self.operation().read().unwrap().region())
    }
    pub fn first_op(&self) -> Result<Arc<RwLock<dyn Op>>> {
        let body_region = self.get_body_region()?;
        let region = match body_region {
            Some(region) => region,
            None => return Err(anyhow::anyhow!("Expected 1 region in module, got 0")),
        };
        let blocks = region.read().unwrap().blocks();
        let block = match blocks.first() {
            Some(block) => block,
            None => return Err(anyhow::anyhow!("Expected 1 block in module, got 0")),
        };
        let ops = block.read().unwrap().ops();
        let ops = ops.read().unwrap();
        let op = ops.first();
        if op.is_none() {
            return Err(anyhow::anyhow!("Expected 1 op, got 0"));
        } else {
            let op = op.unwrap().clone();
            Ok(op)
        }
    }
}
