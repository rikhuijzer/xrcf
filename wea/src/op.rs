use crate::WeaParse;
use anyhow::Result;
use std::fmt::Formatter;
use xrcf::frontend::Parse;
use xrcf::frontend::Parser;
use xrcf::frontend::ParserDispatch;
use xrcf::frontend::TokenKind;
use xrcf::ir::Block;
use xrcf::ir::Op;
use xrcf::ir::Operation;
use xrcf::ir::OperationName;
use xrcf::shared::Shared;

/// The token kind used for variables in ArnoldC.
///
/// In ArnoldC variables are always bare identifiers meaning `x` is a valid
/// variable. For example, percent identifiers like `%x` are not valid.
const TOKEN_KIND: TokenKind = TokenKind::BareIdentifier;

pub enum Visibility {
    Public,
    Private,
}

pub struct FuncOp {
    operation: Shared<Operation>,
    pub visibility: Option<Visibility>,
    pub identifier: Option<String>,
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("fn".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        FuncOp {
            operation,
            visibility: None,
            identifier: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", Self::operation_name())?;
        Ok(())
    }
}

impl Parse for FuncOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let next = parser.peek();
        let visibility = if next.kind == TokenKind::BareIdentifier && next.lexeme == "pub" {
            parser.advance();
            Visibility::Public
        } else {
            Visibility::Private
        };
        parser.parse_operation_name_into::<FuncOp>(&mut operation)?;
        let identifier = parser.expect(TokenKind::BareIdentifier)?;
        parser.expect(TokenKind::LParen)?;
        if parser.peek().kind != TokenKind::RParen {
            parser.parse_wea_function_arguments_into(&mut operation)?;
        }
        parser.expect(TokenKind::RParen)?;
        let operation = Shared::new(operation.into());
        let mut op = FuncOp::new(operation.clone());
        op.visibility = Some(visibility);
        op.identifier = Some(identifier.lexeme);
        let op = Shared::new(op.into());
        Ok(op)
    }
}

/// `+`
pub struct PlusOp {
    operation: Shared<Operation>,
}

impl Op for PlusOp {
    fn operation_name() -> OperationName {
        OperationName::new("+".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        PlusOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", Self::operation_name())?;
        Ok(())
    }
}

impl Parse for PlusOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        parser.parse_operation_name_into::<PlusOp>(&mut operation)?;
        let operation = Shared::new(operation.into());
        let op = PlusOp {
            operation: operation.clone(),
        };
        let op = Shared::new(op.into());
        Ok(op)
    }
}
