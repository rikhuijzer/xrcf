use crate::ir::operation::Operation;
use crate::ir::BlockArgument;
use crate::ir::Op;
use crate::ir::OpResult;
use crate::ir::OperationName;
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
    fn op<T: Parse>(parser: &mut Parser<T>) -> Result<Arc<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_results(parser.results()?);

        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == "arith.constant");
        operation.set_name(ConstantOp::operation_name());
        let operand = match parser.peek().kind {
            TokenKind::Integer => {
                let integer = parser.advance();
                let integer = integer.lexeme.clone();
                let typ = parser.typ()?;
                let argument = BlockArgument::new(integer, typ);
                Value::BlockArgument(argument)
            }
            _ => {
                return Err(anyhow::anyhow!("Expected integer constant"));
            }
        };
        operation.set_operands(Arc::new(vec![operand]));

        let operation = Arc::new(RwLock::new(operation));
        Ok(Arc::new(ConstantOp { operation }))
    }
}

pub struct AddiOp {
    operation: Arc<RwLock<Operation>>,
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
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation().read().unwrap())
    }
}

impl<T: Parse> Parser<T> {
    fn argument(&mut self) -> Result<Value> {
        let identifier = self.expect(TokenKind::PercentIdentifier)?;
        let name = identifier.lexeme.clone();
        let typ = Type::new("any".to_string());
        let value = Value::OpResult(OpResult::new(name, typ));
        Ok(value)
    }
    pub fn arguments(&mut self) -> Result<Vec<Value>> {
        let mut arguments = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            arguments.push(self.argument()?);
            if self.check(TokenKind::Comma) {
                let _comma = self.advance();
            }
        }
        Ok(arguments)
    }
    pub fn results(&mut self) -> Result<Vec<Value>> {
        let mut results = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            results.push(self.argument()?);
            if self.check(TokenKind::Equal) {
                let _equal = self.advance();
            }
        }
        Ok(results)
    }
}

impl Parse for AddiOp {
    fn op<T: Parse>(parser: &mut Parser<T>) -> Result<Arc<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_results(parser.results()?);

        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == "arith.addi");
        operation.set_name(AddiOp::operation_name());
        operation.set_operands(Arc::new(parser.arguments()?));
        let _colon = parser.expect(TokenKind::Colon)?;
        let result_type = parser.expect(TokenKind::IntType)?;
        let result_type = Type::new(result_type.lexeme.clone());
        operation.set_result_types(vec![result_type]);

        let operation = Arc::new(RwLock::new(operation));
        Ok(Arc::new(AddiOp { operation }))
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
