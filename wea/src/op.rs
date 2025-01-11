use crate::WeaParse;
use anyhow::Result;
use std::fmt::Formatter;
use xrcf::frontend::Parse;
use xrcf::frontend::Parser;
use xrcf::frontend::ParserDispatch;
use xrcf::frontend::TokenKind;
use xrcf::ir::display_region_inside_func;
use xrcf::ir::Block;
use xrcf::ir::Op;
use xrcf::ir::Operation;
use xrcf::ir::OperationName;
use xrcf::ir::Prefixes;
use xrcf::ir::Values;
use xrcf::shared::Shared;
use xrcf::shared::SharedExt;

const PREFIXES: Prefixes = Prefixes {
    argument: "arg",
    block: "bb",
    ssa: "",
};

/// Tokens in wea have no prefix unlike MLIR which uses `%` (PercentIdentifier).
const TOKEN_KIND: TokenKind = TokenKind::BareIdentifier;

/// Display function arguments wea style (e.g., `a: i32`)
fn display_function_arguments(f: &mut Formatter<'_>, arguments: &Values) -> std::fmt::Result {
    write!(f, "(")?;
    let arguments = arguments
        .clone()
        .into_iter()
        .map(|argument| {
            let name = argument.rd().name().unwrap();
            let typ = argument.rd().typ().unwrap().rd().to_string();
            format!("{name}: {typ}")
        })
        .collect::<Vec<_>>()
        .join(", ");
    write!(f, "{}", arguments)?;
    write!(f, ")")
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    fn is_func(&self) -> bool {
        true
    }
    fn prefixes(&self) -> Prefixes {
        PREFIXES
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        if self.visibility.clone().expect("visibility not set") == Visibility::Public {
            write!(f, "pub ")?;
        }
        write!(f, "{} ", Self::operation_name())?;
        write!(
            f,
            "{}",
            self.identifier.clone().expect("identifier not set")
        )?;
        display_function_arguments(f, &self.operation.rd().arguments())?;
        let result_type = self.operation.rd().result_type(0);
        if result_type.is_some() {
            write!(f, " -> {}", result_type.unwrap().rd().to_string())?;
        }
        display_region_inside_func(f, &self.operation.rd(), indent)?;
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
        parser.parse_wea_function_arguments_into(&mut operation)?;
        if parser.peek().kind == TokenKind::Arrow {
            parser.advance();
            operation.set_anonymous_result(T::parse_type(parser)?)?;
        }
        let operation = Shared::new(operation.into());
        let mut op = FuncOp::new(operation.clone());
        op.visibility = Some(visibility);
        op.identifier = Some(identifier.lexeme);
        let op = Shared::new(op.into());
        Parser::parse_func_body(parser, op.clone())?;
        Ok(op)
    }
}

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
    fn prefixes(&self) -> Prefixes {
        PREFIXES
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation.clone();
        write!(
            f,
            "{} ",
            operation.rd().operand(0).unwrap().rd().to_string()
        )?;
        write!(f, "{} ", Self::operation_name())?;
        write!(f, "{}", operation.rd().operand(1).unwrap().rd().to_string())?;
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
        let parent = parent.expect("no parent");
        let lhs = parser.parse_op_operand(parent.clone(), TOKEN_KIND)?;
        operation.operands.vec().wr().push(lhs);
        parser.expect(TokenKind::Plus)?;
        let rhs = parser.parse_op_operand(parent, TOKEN_KIND)?;
        operation.operands.vec().wr().push(rhs);

        let operation = Shared::new(operation.into());
        let op = PlusOp {
            operation: operation.clone(),
        };
        let op = Shared::new(op.into());
        Ok(op)
    }
}
