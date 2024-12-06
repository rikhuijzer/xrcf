use crate::ir::Block;
use crate::ir::GuardedOperation;
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
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

const TOKEN_KIND: TokenKind = TokenKind::PercentIdentifier;

/// `scf.if`
///
/// ```ebnf
/// `scf.if` $condition `then` `{` $then `}` `else` `{` $else `}`
/// ```
pub struct IfOp {
    operation: Arc<RwLock<Operation>>,
    then: Option<Arc<RwLock<Region>>>,
    els: Option<Arc<RwLock<Region>>>,
}

impl IfOp {
    pub fn els(&self) -> Option<Arc<RwLock<Region>>> {
        self.els.clone()
    }
    pub fn then(&self) -> Option<Arc<RwLock<Region>>> {
        self.then.clone()
    }
    pub fn set_els(&mut self, els: Option<Arc<RwLock<Region>>>) {
        self.els = els;
    }
    pub fn set_then(&mut self, then: Option<Arc<RwLock<Region>>>) {
        self.then = then;
    }
}

impl Op for IfOp {
    fn operation_name() -> OperationName {
        OperationName::new("scf.if".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        IfOp {
            operation,
            then: None,
            els: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn ops(&self) -> Vec<Arc<RwLock<dyn Op>>> {
        let mut ops = vec![];
        if let Some(then) = self.then() {
            ops.extend(then.ops());
        }
        if let Some(els) = self.els() {
            ops.extend(els.ops());
        }
        ops
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let has_results = !self.operation.results().is_empty();
        if has_results {
            write!(f, "{} = ", self.operation.results())?;
        }
        write!(f, "{} ", self.operation.name())?;
        write!(f, "{}", self.operation.operands())?;
        if has_results {
            write!(f, " -> ({})", self.operation.results().types())?;
        }
        let then = self.then.clone().expect("Expected `then` region");
        let then = then.try_read().unwrap();
        then.display(f, indent)?;
        let els = self.els.clone().expect("Expected `else` region");
        let els = els.try_read().unwrap();
        write!(f, " else")?;
        els.display(f, indent)?;
        Ok(())
    }
}

impl Parse for IfOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
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

        let operation = Arc::new(RwLock::new(operation));
        let op = IfOp {
            operation,
            then: None,
            els: None,
        };
        let op = Arc::new(RwLock::new(op));
        let then = parser.parse_region(op.clone())?;
        let else_keyword = parser.expect(TokenKind::BareIdentifier)?;
        if else_keyword.lexeme() != "else" {
            return Err(anyhow::anyhow!("Expected `else`, but got {}", else_keyword));
        }
        let els = parser.parse_region(op.clone())?;
        let op_write = op.clone();
        let mut op_write = op_write.try_write().unwrap();
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
    operation: Arc<RwLock<Operation>>,
}

impl Op for YieldOp {
    fn operation_name() -> OperationName {
        OperationName::new("scf.yield".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        YieldOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} ", self.operation.name())?;
        self.operation.operands().display_with_types(f)?;
        Ok(())
    }
}

impl Parse for YieldOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        parser.parse_operation_name_into::<YieldOp>(&mut operation)?;
        let parent = parent.expect("Expected parent");
        let operand = parser.parse_op_operand_into(parent, TOKEN_KIND, &mut operation)?;
        parser.expect(TokenKind::Colon)?;
        parser.parse_type_for_op_operand(operand)?;

        let operation = Arc::new(RwLock::new(operation));
        let op = YieldOp { operation };
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
}
