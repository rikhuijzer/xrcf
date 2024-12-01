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
    thn: Option<Arc<RwLock<Block>>>,
    els: Option<Arc<RwLock<Block>>>,
}

impl IfOp {}

impl Op for IfOp {
    fn operation_name() -> OperationName {
        OperationName::new("scf.if".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        IfOp {
            operation,
            thn: None,
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
        todo!()
    }
}
