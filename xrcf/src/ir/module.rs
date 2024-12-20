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
use crate::shared::Shared;
use crate::shared::SharedExt;
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
        write!(f, "{}", crate::ir::spaces(indent))?;
        self.operation().rd().display(f, indent)
    }
}

impl Display for ModuleOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.operation().rd())
    }
}

impl ModuleOp {
    pub fn get_body_region(&self) -> Result<Option<Arc<RwLock<Region>>>> {
        Ok(self.operation().rd().region())
    }
    pub fn first_op(&self) -> Result<Arc<RwLock<dyn Op>>> {
        let region = match self.get_body_region()? {
            Some(region) => region,
            None => return Err(anyhow::anyhow!("Expected 1 region in module, got 0")),
        };
        let block = match region.blocks().into_iter().next() {
            Some(block) => block,
            None => return Err(anyhow::anyhow!("Expected 1 block in module, got 0")),
        };
        let x = match block.rd().ops().rd().first() {
            None => {
                return Err(anyhow::anyhow!("Expected 1 op, got 0"));
            }
            Some(op) => Ok(op.clone()),
        };
        x
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
        let operation = Shared::new(operation.into());
        let op = ModuleOp {
            operation: operation.clone(),
        };
        let op = Shared::new(op.into());

        let region = parser.parse_region(op.clone());
        let mut operation = operation.wr();
        operation.set_region(Some(region?));
        operation.set_parent(parent.clone());

        Ok(op)
    }
}
