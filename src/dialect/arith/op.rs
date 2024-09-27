use crate::canonicalize::CanonicalizeResult;
use crate::ir::operation::Operation;
use crate::ir::Block;
use crate::ir::BlockArgument;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::OpResult;
use crate::ir::OperationName;
use crate::ir::OperationOperands;
use crate::ir::Type;
use crate::ir::Value;
use crate::parser::Parse;
use crate::parser::Parser;
use crate::parser::TokenKind;
use crate::Dialect;
use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

struct Arith {}

pub struct ConstantOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for ConstantOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.constant".to_string())
    }
    fn from_operation(_operation: Arc<RwLock<Operation>>) -> Result<Self> {
        todo!()
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation().read().unwrap())
    }
}

impl<T: Parse> Parser<T> {
    /// Parse a type definition (e.g., `: i64`).
    fn typ(&mut self) -> Result<Type> {
        let _colon = self.expect(TokenKind::Colon)?;
        let typ = self.expect(TokenKind::IntType)?;
        let typ = Type::new(typ.lexeme.clone());
        Ok(typ)
    }
}

impl Parse for ConstantOp {
    fn op<T: Parse>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_results(parser.results()?);

        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == "arith.constant");
        operation.set_name(ConstantOp::operation_name());
        let operand: OpOperand = match parser.peek().kind {
            TokenKind::Integer => {
                let integer = parser.advance();
                let integer = integer.lexeme.clone();
                let typ = parser.typ()?;
                // Determine value.
                todo!();
                // let operand = OpOperand::new(
                // Value::BlockArgument(argument)
            }
            _ => {
                return Err(anyhow::anyhow!("Expected integer constant"));
            }
        };
        let operand = Arc::new(RwLock::new(operand));
        operation.set_operands(Arc::new(RwLock::new(vec![operand])));
        operation.set_parent(parent);
        let operation = Arc::new(RwLock::new(operation));
        Ok(Arc::new(RwLock::new(ConstantOp { operation })))
    }
}

pub struct AddiOp {
    operation: Arc<RwLock<Operation>>,
}

impl AddiOp {
    /// Canonicalize `addi(addi(x, c0), c1) -> addi(x, c0 + c1)`.
    fn addi_add_constant(&mut self) -> CanonicalizeResult {
        let operands = self.operation().read().unwrap().operands();
        let operands = operands.read().unwrap();
        assert!(operands.len() == 2);
        let lhs = &operands[0];
        let rhs = &operands[1];
        CanonicalizeResult::Changed
    }
}

impl Op for AddiOp {
    fn operation_name() -> OperationName {
        OperationName::new("arith.addi".to_string())
    }
    fn from_operation(_operation: Arc<RwLock<Operation>>) -> Result<Self> {
        todo!()
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn canonicalize(&mut self) -> CanonicalizeResult {
        self.addi_add_constant()
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation().read().unwrap())
    }
}

impl<T: Parse> Parser<T> {
    fn operand(&mut self, parent: Arc<RwLock<Block>>) -> Result<Arc<RwLock<OpOperand>>> {
        let identifier = self.expect(TokenKind::PercentIdentifier)?;
        let name = identifier.lexeme.clone();
        let block = parent.read().unwrap();
        let assignment = block.assignment(name.clone());
        let assignment = match assignment {
            Some(assignment) => assignment,
            None => {
                let msg = "Expected assignment before use.";
                let msg = self.error(&identifier, msg);
                return Err(anyhow::anyhow!(msg));
            }
        };
        // let typ = Type::new("any".to_string());
        // let value = Value::OpResult(OpResult::new(name, typ));
        // let operand = OpOperand::new(value, name);
        // Ok(Arc::new(operand))
    }
    pub fn operands(&mut self, parent: Arc<RwLock<Block>>) -> Result<OperationOperands> {
        let mut arguments = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            let operand = self.operand(parent.clone())?;
            arguments.push(operand);
            if self.check(TokenKind::Comma) {
                let _comma = self.advance();
            }
        }
        Ok(Arc::new(RwLock::new(arguments)))
    }
    pub fn results(&mut self) -> Result<Vec<Arc<Value>>> {
        let mut results = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            let identifier = self.expect(TokenKind::PercentIdentifier)?;
            let name = identifier.lexeme.clone();
            let typ = Type::new("any".to_string());
            let result = Value::OpResult(OpResult::new(name, typ));
            results.push(Arc::new(result));
            if self.check(TokenKind::Equal) {
                let _equal = self.advance();
            }
        }
        Ok(results)
    }
}

impl Parse for AddiOp {
    fn op<T: Parse>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_results(parser.results()?);

        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == "arith.addi");
        operation.set_name(AddiOp::operation_name());
        assert!(parent.is_some());
        operation.set_parent(parent.clone());
        operation.set_operands(parser.operands(parent.unwrap())?);
        let _colon = parser.expect(TokenKind::Colon)?;
        let result_type = parser.expect(TokenKind::IntType)?;
        let result_type = Type::new(result_type.lexeme.clone());
        operation.set_result_types(vec![result_type]);

        let operation = Arc::new(RwLock::new(operation));
        Ok(Arc::new(RwLock::new(AddiOp { operation })))
    }
}

// In MLIR this works by taking an OpAsmParser and parsing
// the elements of the op.
// Parsing tries to cast the elements to the expected types.
// If all succeeds, the elements are parsed into the struct.
// todo!()
// }
// enum ArithOp {
//    Addi(Addi),
//}

impl Dialect for Arith {
    fn name(&self) -> &'static str {
        "arith"
    }

    fn description(&self) -> &'static str {
        "Arithmetic operations."
    }

    // Probably we don't want to have a global obs state but instead
    // have some differrent implementations for common functions.
    // fn ops(&self) ->
    // }
}
