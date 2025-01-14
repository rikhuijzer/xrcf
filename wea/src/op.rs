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
use xrcf::ir::OpResult;
use xrcf::ir::Operation;
use xrcf::ir::OperationName;
use xrcf::ir::Prefixes;
use xrcf::ir::UnsetOpResults;
use xrcf::ir::Value;
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

pub struct BinaryExpression {
    operation: Shared<Operation>,
    left: Option<Shared<dyn Op>>,
    right: Option<Shared<dyn Op>>,
}

impl Op for BinaryExpression {
    fn operation_name() -> OperationName {
        OperationName::new("binary".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        BinaryExpression {
            operation,
            left: None,
            right: None,
        }
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn prefixes(&self) -> Prefixes {
        PREFIXES
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} ", Self::operation_name().name())?;
        write!(f, "{}", self.left.as_ref().unwrap().rd())?;
        write!(f, "{}", self.right.as_ref().unwrap().rd())?;
        Ok(())
    }
}

enum Expr {
    Leaf(String),
    Unary(String),
}

fn parse_expr_inner<T: ParserDispatch>(
    parser: &mut Parser<T>,
    parent: Option<Shared<Block>>,
) -> Result<Expr> {
    let next = parser.peek();
    match next.kind {
        TokenKind::BareIdentifier => {
            return Ok(Expr::Leaf(next.lexeme.clone()));
        }
        TokenKind::LParen => {
            let expr = parse_expr_outer(parser, None)?;
            parser.expect(TokenKind::RParen)?;
            Ok(expr)
        }
        _ => {
            return Err(anyhow::anyhow!("expected identifier or lparen"));
        }
    }
}

/// Parse an expression.
///
/// Based on https://www.scattered-thoughts.net/writing/better-operator-precedence/.
fn parse_expr_outer<T: ParserDispatch>(
    parser: &mut Parser<T>,
    parent: Option<Shared<Block>>,
) -> Result<Expr> {
    let left = parse_expr_inner(parser, parent)?;

    todo!()
}

impl Parse for BinaryExpression {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let expr = parse_expr_outer(parser, None);
        todo!()
    }
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
            write!(f, " {}", result_type.unwrap().rd())?;
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
        operation.set_anonymous_result(T::parse_type(parser)?)?;
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
        write!(f, "{} ", operation.rd().operand(0).unwrap().rd())?;
        write!(f, "{} ", Self::operation_name())?;
        write!(f, "{}", operation.rd().operand(1).unwrap().rd())?;
        Ok(())
    }
}

/// Add a result to the operation when no result was defined.
fn add_implicit_result_into(operation: &mut Operation) -> Result<UnsetOpResults> {
    let op_result = Value::OpResult(OpResult::default());
    let value = Shared::new(op_result.into());
    let results = Values::from_vec(vec![value.clone()]);
    operation.set_results(results.clone());
    Ok(UnsetOpResults::new(results))
}

impl Parse for PlusOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let results = if parser.defines_result() {
            parser.parse_op_results_into(TOKEN_KIND, &mut operation)?
        } else {
            add_implicit_result_into(&mut operation)?
        };
        let parent = parent.expect("no parent");
        let lhs = parser.parse_op_operand(parent.clone(), TOKEN_KIND)?;
        let typ = lhs.rd().typ().expect("no type");
        operation.operands.vec().wr().push(lhs);
        parser.expect(TokenKind::Plus)?;
        let rhs = parser.parse_op_operand(parent, TOKEN_KIND)?;
        operation.operands.vec().wr().push(rhs);

        let operation = Shared::new(operation.into());
        let op = PlusOp {
            operation: operation.clone(),
        };
        let op = Shared::new(op.into());
        results.set_defining_op(op.clone());
        results.set_types(vec![typ]);
        Ok(op)
    }
}
