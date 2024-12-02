use crate::convert::ChangedOp;
use crate::convert::RewriteResult;
use crate::ir::AnyType;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::GuardedOpOperand;
use crate::ir::GuardedOperation;
use crate::ir::GuardedValue;
use crate::ir::IntegerAttr;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::OpResult;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Value;
use crate::ir::Region;
use crate::ir::Values;
use crate::parser::Parse;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

const TOKEN_KIND: TokenKind = TokenKind::PercentIdentifier;

pub struct IfOp {
    operation: Arc<RwLock<Operation>>,
    then: Option<Arc<RwLock<Region>>>,
    els: Option<Arc<RwLock<Region>>>,
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
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} = ", self.operation.results())?;
        write!(f, "{}", self.operation.name())?;
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
        parser.expect(TokenKind::Equal)?;
        parser.parse_operation_name_into::<IfOp>(&mut operation)?;

        let parent = parent.expect("Expected parent");
        let _condition = parser.parse_op_operand_into(
            parent.clone(), TokenKind::PercentIdentifier, &mut operation)?;

        if parser.check(TokenKind::Arrow) {
            parser.advance();
            parser.expect(TokenKind::LParen)?;
            let return_types = parser.parse_types()?;
            results.set_types(return_types);
            parser.expect(TokenKind::RParen)?;
        }

        let operation = Arc::new(RwLock::new(operation));
        let op = IfOp { operation, then: None, els: None };
        let op = Arc::new(RwLock::new(op));
        let then = parser.parse_region(op.clone())?;
        let els = parser.parse_region(op.clone())?;
        let op_write = op.clone();
        let mut op_write = op_write.try_write().unwrap();
        op_write.then = Some(then);
        op_write.els = Some(els);
        results.set_defining_op(op.clone());
        Ok(op)
    }
}
