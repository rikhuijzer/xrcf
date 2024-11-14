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

pub struct CondBranchOp {
    operation: Arc<RwLock<Operation>>,
}

impl CondBranchOp {}

impl Op for CondBranchOp {
    fn operation_name() -> OperationName {
        OperationName::new("cf.cond_br".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        CondBranchOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_const(&self) -> bool {
        false
    }
    fn is_pure(&self) -> bool {
        true
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
}

impl Parse for CondBranchOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        parser.parse_operation_name_into::<CondBranchOp>(&mut operation)?;

        let operation = Arc::new(RwLock::new(operation));
        let op = CondBranchOp { operation };
        let op = Arc::new(RwLock::new(op));
        todo!()
    }
}

impl Display for CondBranchOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.operation.read().unwrap())
    }
}
