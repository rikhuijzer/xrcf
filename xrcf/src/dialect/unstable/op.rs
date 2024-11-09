use crate::ir::Block;
use crate::ir::Constant;
use crate::ir::GuardedOperation;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::StringAttr;
use crate::ir::Value;
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
}

impl PrintfOp {
    pub fn text(&self) -> StringAttr {
        let operands = self.operation.operand(0);
        let operand = operands.expect("no operand");
        let operand = operand.try_read().unwrap();
        let value = operand.value();
        let value = value.try_read().unwrap();
        let text = match &*value {
            Value::Constant(constant) => constant,
            _ => panic!("expected constant"),
        };
        let text = text.value();
        let text = text.as_any().downcast_ref::<StringAttr>().unwrap();
        text.clone()
    }
    pub fn set_text(&mut self, text: StringAttr) {
        let value = Constant::new(Arc::new(text));
        let value = Value::Constant(value);
        let value = Arc::new(RwLock::new(value));
        let operand = OpOperand::new(value);
        let operand = Arc::new(RwLock::new(operand));
        self.operation.set_operand(0, operand);
    }
}

impl Op for PrintfOp {
    fn operation_name() -> OperationName {
        OperationName::new("unstable.printf".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        PrintfOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation().name())?;
        write!(f, "({})", self.operation().operands())?;
        Ok(())
    }
}

impl Parse for PrintfOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        parser.parse_operation_name_into::<PrintfOp>(&mut operation)?;
        operation.set_parent(parent.clone());

        parser.expect(TokenKind::LParen)?;
        parser.parse_op_operands_into(parent.expect("no parent"), &mut operation)?;
        parser.expect(TokenKind::RParen)?;
        let operation = Arc::new(RwLock::new(operation));
        let op = PrintfOp { operation };

        Ok(Arc::new(RwLock::new(op)))
    }
}
