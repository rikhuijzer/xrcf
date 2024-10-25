use crate::ir::Block;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::StringAttr;
use crate::parser::Parse;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// `unstable.printf`
///
/// This operation is lowered to LLVM IR by default, so might not be usable on
/// all platforms.
pub struct PrintfOp {
    operation: Arc<RwLock<Operation>>,
    text: Option<StringAttr>,
}

impl PrintfOp {
    pub fn text(&self) -> Option<StringAttr> {
        self.text.clone()
    }
}

impl Op for PrintfOp {
    fn operation_name() -> OperationName {
        OperationName::new("unstable.printf".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        PrintfOp {
            operation,
            text: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation().read().unwrap())
    }
}

impl Parse for PrintfOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        parser.parse_operation_name_into::<PrintfOp>(&mut operation)?;
        operation.set_parent(parent);

        let operation = Arc::new(RwLock::new(operation));
        parser.expect(TokenKind::LParen)?;
        let text = parser.parse_string()?;
        parser.expect(TokenKind::RParen)?;
        let text = Some(text);
        let op = PrintfOp { operation, text };

        Ok(Arc::new(RwLock::new(op)))
    }
}
