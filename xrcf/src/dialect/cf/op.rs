use crate::ir::Block;
use crate::ir::BlockDest;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
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

/// `cf.cond_br`
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
        parser.parse_op_operands_into(parent.clone().unwrap(), TOKEN_KIND, &mut operation)?;

        let operation = Arc::new(RwLock::new(operation));
        let op = CondBranchOp { operation };
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
}

impl Display for CondBranchOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

/// `cf.br`
///
/// ```ebnf
/// `cf.br` $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
/// ```
pub struct BranchOp {
    operation: Arc<RwLock<Operation>>,
    dest: Option<Arc<RwLock<BlockDest>>>,
}

impl BranchOp {
    pub fn dest(&self) -> Option<Arc<RwLock<BlockDest>>> {
        self.dest.clone()
    }
}

impl Op for BranchOp {
    fn operation_name() -> OperationName {
        OperationName::new("cf.br".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        BranchOp {
            operation,
            dest: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_pure(&self) -> bool {
        true
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} ", self.operation.try_read().unwrap())?;
        let dest = self.dest.as_ref().unwrap();
        let dest = dest.try_read().unwrap();
        write!(f, "{}", dest)
    }
}

impl Parse for BranchOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        parser.parse_operation_name_into::<BranchOp>(&mut operation)?;

        let operation = Arc::new(RwLock::new(operation));
        let dest = parser.parse_block_dest()?;
        let dest = Some(Arc::new(RwLock::new(dest)));
        let op = BranchOp { operation, dest };
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
}

impl Display for BranchOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
