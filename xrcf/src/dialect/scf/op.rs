use crate::frontend::Parse;
use crate::frontend::Parser;
use crate::frontend::ParserDispatch;
use crate::frontend::TokenKind;
use crate::ir::Block;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Region;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use std::fmt::Formatter;

const TOKEN_KIND: TokenKind = TokenKind::PercentIdentifier;

/// `scf.if`
///
/// ```ebnf
/// `scf.if` $condition `then` `{` $then `}` `else` `{` $else `}`
/// ```
pub struct IfOp {
    operation: Shared<Operation>,
    then: Option<Shared<Region>>,
    els: Option<Shared<Region>>,
}

impl IfOp {
    pub fn els(&self) -> Option<Shared<Region>> {
        self.els.clone()
    }
    pub fn then(&self) -> Option<Shared<Region>> {
        self.then.clone()
    }
    pub fn set_els(&mut self, els: Option<Shared<Region>>) {
        self.els = els;
    }
    pub fn set_then(&mut self, then: Option<Shared<Region>>) {
        self.then = then;
    }
}

impl Op for IfOp {
    fn operation_name() -> OperationName {
        OperationName::new("scf.if".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        IfOp {
            operation,
            then: None,
            els: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn ops(&self) -> Vec<Shared<dyn Op>> {
        let mut ops = vec![];
        if let Some(then) = self.then() {
            ops.extend(then.rd().ops());
        }
        if let Some(els) = self.els() {
            ops.extend(els.rd().ops());
        }
        ops
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let has_results = !self.operation.rd().results().is_empty();
        if has_results {
            write!(f, "{} = ", self.operation.rd().results())?;
        }
        write!(f, "{} ", self.operation.rd().name())?;
        write!(f, "{}", self.operation.rd().operands())?;
        if has_results {
            write!(f, " -> ({})", self.operation.rd().results().types())?;
        }
        self.then().expect("no region").rd().display(f, indent)?;
        write!(f, " else")?;
        self.els().expect("no region").rd().display(f, indent)?;
        Ok(())
    }
}

impl Parse for IfOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let results = parser.parse_op_results_into(TOKEN_KIND, &mut operation)?;
        let has_results = !results.values().is_empty();
        if has_results {
            parser.expect(TokenKind::Equal)?;
        }
        parser.parse_operation_name_into::<IfOp>(&mut operation)?;
        let parent = parent.expect("Expected parent");
        let _condition = parser.parse_op_operand_into(
            parent.clone(),
            TokenKind::PercentIdentifier,
            &mut operation,
        )?;

        if has_results {
            parser.expect(TokenKind::Arrow)?;
            parser.expect(TokenKind::LParen)?;
            let return_types = parser.parse_types()?;
            results.set_types(return_types);
            parser.expect(TokenKind::RParen)?;
        }

        let operation = Shared::new(operation.into());
        let op = IfOp {
            operation,
            then: None,
            els: None,
        };
        let op = Shared::new(op.into());
        let then = parser.parse_region(op.clone())?;
        let else_keyword = parser.expect(TokenKind::BareIdentifier)?;
        if else_keyword.lexeme() != "else" {
            return Err(anyhow::anyhow!("Expected `else`, but got {}", else_keyword));
        }
        let els = parser.parse_region(op.clone())?;
        let op_write = op.clone();
        let mut op_write = op_write.wr();
        op_write.then = Some(then);
        op_write.els = Some(els);
        results.set_defining_op(op.clone());
        Ok(op)
    }
}

/// `scf.yield`
///
/// ```ebnf
/// `scf.yield` $operands `:` type($operands)
/// ```
/// For example,
/// ```mlir
/// scf.yield %0 : i32
/// ```
pub struct YieldOp {
    operation: Shared<Operation>,
}

impl Op for YieldOp {
    fn operation_name() -> OperationName {
        OperationName::new("scf.yield".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        YieldOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} ", self.operation.rd().name())?;
        self.operation.rd().operands().display_with_types(f)?;
        Ok(())
    }
}

impl Parse for YieldOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        parser.parse_operation_name_into::<YieldOp>(&mut operation)?;
        let parent = parent.expect("Expected parent");
        let operand = parser.parse_op_operand_into(parent, TOKEN_KIND, &mut operation)?;
        parser.expect(TokenKind::Colon)?;
        parser.parse_type_for_op_operand(operand)?;

        let operation = Shared::new(operation.into());
        let op = YieldOp { operation };
        let op = Shared::new(op.into());
        Ok(op)
    }
}
