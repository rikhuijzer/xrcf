use crate::frontend::Parse;
use crate::frontend::Parser;
use crate::frontend::ParserDispatch;
use crate::frontend::TokenKind;
use crate::ir::Block;
use crate::ir::Constant;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::StringAttr;
use crate::ir::Value;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;

const TOKEN_KIND: TokenKind = TokenKind::PercentIdentifier;

/// `experimental.printf`
///
/// This operation is lowered to LLVM IR by default, so might not be usable on
/// all platforms.
pub struct PrintfOp {
    operation: Shared<Operation>,
}

impl PrintfOp {
    pub fn text(&self) -> StringAttr {
        let operands = self.operation.rd().operand(0);
        let operand = operands.expect("no operand");
        let value = &operand.rd().value;
        let value = value.rd();
        match &*value {
            Value::Constant(constant) => constant
                .value()
                .as_any()
                .downcast_ref::<StringAttr>()
                .unwrap()
                .clone(),
            _ => panic!("expected constant"),
        }
    }
    /// Set the first operand to `printf`.
    pub fn set_text(&mut self, text: StringAttr) {
        let value = Constant::new(Arc::new(text));
        let value = Value::Constant(value);
        let value = Shared::new(value.into());
        let operand = OpOperand::new(value);
        let operand = Shared::new(operand.into());
        self.operation.wr().set_operand(0, operand);
    }
}

impl Op for PrintfOp {
    fn operation_name() -> OperationName {
        OperationName::new("experimental.printf".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        PrintfOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation().rd().name())?;
        write!(f, "({})", self.operation().rd().operands())?;
        Ok(())
    }
}

impl Parse for PrintfOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        parser.parse_operation_name_into::<PrintfOp>(&mut operation)?;
        operation.set_parent(parent.clone());

        parser.expect(TokenKind::LParen)?;
        parser.parse_op_operands_into(parent.expect("no parent"), TOKEN_KIND, &mut operation)?;
        parser.expect(TokenKind::RParen)?;
        let operation = Shared::new(operation.into());
        let op = PrintfOp { operation };

        Ok(Shared::new(op.into()))
    }
}
