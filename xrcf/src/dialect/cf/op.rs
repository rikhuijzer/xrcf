use crate::ir::Block;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
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
        parser.parse_op_operands_into(parent.clone().unwrap(), TOKEN_KIND, &mut operation)?;

        let operation = Arc::new(RwLock::new(operation));
        let op = CondBranchOp { operation };
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
}

impl Display for CondBranchOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.operation.try_read().unwrap())
    }
}

/// Call to a destination of type block.
///
/// For example, `^merge(%c4: i32)` in:
///
/// ```mlir
/// cr.br ^merge(%c4: i32)
/// ```
///
/// This data structure is needed since ops such as `cf.cond_br` can have
/// multiple destinations.
pub struct BlockDest {
    name: String,
    operands: Values,
}

impl BlockDest {
    /// The name of the destination block (e.g., `^merge`).
    pub fn name(&self) -> String {
        self.name.clone()
    }
    /// The operands of the destination block (e.g., `(%c4: i32)`).
    pub fn operands(&self) -> Values {
        self.operands.clone()
    }
    /// Parse `^merge(%c4: i32)` when it represents a block destination.
    ///
    /// Example:
    /// ```mlir
    /// cr.br ^merge(%c4: i32)
    /// ```
    pub fn parse<T: ParserDispatch>(parser: &mut Parser<T>) -> Result<BlockDest> {
        let name = parser.expect(TokenKind::CaretIdentifier)?;
        let name = name.lexeme.clone();
        let operands = parser.parse_function_arguments()?;
        Ok(BlockDest { name, operands })
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

impl Parse for BranchOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        parser.parse_operation_name_into::<BranchOp>(&mut operation)?;

        let operation = Arc::new(RwLock::new(operation));
        let dest = BlockDest::parse(parser)?;
        let dest = Some(Arc::new(RwLock::new(dest)));
        let op = BranchOp { operation, dest };
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
}

impl Display for BranchOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.operation.try_read().unwrap())
    }
}
