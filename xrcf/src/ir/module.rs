use crate::ir::Block;
use crate::ir::GuardedRegion;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Region;
use crate::parser::Parse;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
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
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        Self { operation }
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
        operation.display(f, indent)
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
        let blocks = region.blocks();
        let blocks = blocks.vec();
        let blocks = blocks.try_read().unwrap();
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

impl Parse for ModuleOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == "module");
        operation.set_name(ModuleOp::operation_name());
        let operation = Arc::new(RwLock::new(operation));
        let op = ModuleOp {
            operation: operation.clone(),
        };
        let op = Arc::new(RwLock::new(op));

        let region = parser.parse_region(op.clone());
        let mut operation = operation.try_write().unwrap();
        operation.set_region(Some(region?));
        operation.set_parent(parent.clone());

        Ok(op)
    }
}
