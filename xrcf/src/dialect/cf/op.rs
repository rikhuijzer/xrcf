use crate::ir::Block;
use crate::ir::GuardedOperation;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::parser::Parse;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

const TOKEN_KIND: TokenKind = TokenKind::PercentIdentifier;

/// `cf.br`
///
/// ```ebnf
/// `cf.br` $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
/// ```
pub struct BranchOp {
    operation: Arc<RwLock<Operation>>,
}

impl BranchOp {
    pub fn dest(&self) -> Option<Arc<RwLock<OpOperand>>> {
        self.operation().operand(0)
    }
    pub fn set_dest(&mut self, dest: Arc<RwLock<OpOperand>>) {
        self.operation().set_operand(0, dest);
    }
}

impl Op for BranchOp {
    fn operation_name() -> OperationName {
        OperationName::new("cf.br".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        BranchOp { operation }
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
        write!(f, "{} ", self.operation.name())?;
        write!(f, "{}", self.dest().expect("dest not set").rd())?;
        let operands = self.operation().operands().into_iter().skip(1);
        if 0 < operands.len() {
            write!(f, "(")?;
            for operand in operands {
                operand.rd().display_with_type(f)?;
            }
            write!(f, ")")?;
        }
        Ok(())
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
        let dest = parser.parse_block_dest()?;
        let dest = Shared::new(dest.into());
        operation.set_operand(0, dest);

        let operation = Shared::new(operation.into());

        if parser.check(TokenKind::LParen) {
            parser.expect(TokenKind::LParen)?;
            let operands = operation.operands().vec();
            let mut operands = operands.wr();
            loop {
                let operand = parser.parse_op_operand(parent.clone().unwrap(), TOKEN_KIND)?;
                operands.push(operand.clone());
                parser.expect(TokenKind::Colon)?;
                let _typ = parser.advance();
                if !parser.check(TokenKind::Comma) {
                    break;
                }
            }
            parser.expect(TokenKind::RParen)?;
        }

        let op = BranchOp { operation };
        let op = Shared::new(op.into());
        Ok(op)
    }
}

impl Display for BranchOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

/// `cf.cond_br`
pub struct CondBranchOp {
    operation: Arc<RwLock<Operation>>,
}

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

        let operation = Shared::new(operation.into());
        let op = CondBranchOp { operation };
        let op = Shared::new(op.into());
        Ok(op)
    }
}

impl Display for CondBranchOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
