use crate::ir::BlockArgument;
use crate::ir::Op;
use crate::ir::OpResult;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Type;
use crate::ir::Value;
use crate::parser::TokenKind;
use crate::Parse;
use crate::Parser;
use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// Note that the operands of the function are internally
/// represented by `BlockArgument`s, but the textual form is inline.
pub struct FuncOp {
    identifier: String,
    operation: Arc<RwLock<Operation>>,
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("func.func".to_string())
    }
    fn from_operation(_operation: Arc<RwLock<Operation>>) -> Result<Self> {
        todo!()
        // Ok(FuncOp { operation })
    }
    fn set_indent(&self, indent: i32) {
        let mut operation = self.operation.write().unwrap();
        operation.set_indent(indent);
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        write!(f, "func.func {}(", self.identifier)?;
        let joined = self
            .operation()
            .read()
            .unwrap()
            .operands()
            .iter()
            .map(|o| o.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{}", joined)?;
        write!(f, ")")?;
        let operation = self.operation();
        if !operation.read().unwrap().result_types().is_empty() {
            let operation = operation.read().unwrap();
            let result_types = operation.result_types();
            if result_types.len() == 1 {
                write!(f, " -> {}", result_types.get(0).unwrap())?;
            } else {
                write!(
                    f,
                    " -> ({})",
                    result_types
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                )?;
            }
        }
        let region = self.operation().read().unwrap().region();
        let region = region.read().unwrap();
        region.display(f, indent + 1)?;
        Ok(())
    }
}

impl<T: Parse> Parser<T> {
    pub fn identifier(&mut self, kind: TokenKind) -> Result<String> {
        let identifier = self.advance();
        if identifier.kind != kind {
            return Err(anyhow::anyhow!(
                "Expected {:?}, got {:?}",
                kind,
                identifier.kind
            ));
        }
        Ok(identifier.lexeme.clone())
    }
    /// %arg0 : i64
    pub fn block_argument(&mut self) -> Result<Value> {
        let identifier = self.expect(TokenKind::PercentIdentifier)?;
        let name = identifier.lexeme.clone();
        let typ = if self.check(TokenKind::Colon) {
            self.advance();
            let typ = self.advance();
            Type::new(typ.lexeme.clone())
        } else {
            Type::new("any".to_string())
        };
        let arg = BlockArgument::new(name, typ);
        let operand: Value = Value::BlockArgument(arg);
        if self.check(TokenKind::Comma) {
            self.advance();
        }
        Ok(operand)
    }
    /// Parse operands:
    /// %arg0 : i64, %arg1 : i64
    pub fn block_arguments(&mut self) -> Result<Vec<Value>> {
        let mut operands = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            operands.push(self.block_argument()?);
        }
        if self.check(TokenKind::RParen) {
            let _rparen = self.advance();
        } else if self.check(TokenKind::Colon) {
        } else {
            return Err(anyhow::anyhow!("Expected ')', got {:?}", self.peek().kind));
        }
        Ok(operands)
    }
    pub fn result_types(&mut self) -> Result<Vec<Type>> {
        let mut result_types = vec![];
        if !self.check(TokenKind::Arrow) {
            return Ok(vec![]);
        } else {
            let _arrow = self.advance();
            while self.check(TokenKind::IntType) {
                let typ = self.advance();
                let typ = Type::new(typ.lexeme.clone());
                result_types.push(typ);
            }
        }
        Ok(result_types)
    }
}

impl Parse for FuncOp {
    fn op<T: Parse>(parser: &mut Parser<T>, indent: i32) -> Result<Arc<dyn Op>> {
        // Similar to `FuncOp::parse` in MLIR's `FuncOps.cpp`.
        let result = if parser.peek_n(1).kind == TokenKind::Equal {
            let result = parser.advance().lexeme.clone();
            let result: Value =
                Value::OpResult(OpResult::new(result, Type::new("any".to_string())));
            Some(result)
        } else {
            None
        };
        let _operation_name = parser.advance();
        let identifier = parser.identifier(TokenKind::AtIdentifier).unwrap();
        if !parser.check(TokenKind::LParen) {
            return Err(anyhow::anyhow!(
                "Expected '(', got {:?}",
                parser.peek().kind
            ));
        }
        let _lparen = parser.advance();
        let mut operation = Operation::default();
        operation.set_operands(Arc::new(parser.block_arguments()?));
        operation.set_result_types(parser.result_types()?);
        operation.set_region(parser.region()?);
        operation.set_indent(indent);
        if let Some(result) = result {
            operation.set_results(vec![result]);
        }
        let operation = Arc::new(RwLock::new(operation));

        let op = FuncOp {
            identifier,
            operation,
        };
        Ok(Arc::new(op))
    }
}

pub struct ReturnOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("return".to_string())
    }
    fn from_operation(operation: Arc<RwLock<Operation>>) -> Result<Self> {
        Ok(ReturnOp { operation })
    }
    fn set_indent(&self, indent: i32) {
        let mut operation = self.operation.write().unwrap();
        operation.set_indent(indent);
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation();
        let operation = operation.read().unwrap();
        let operands = operation.operands().clone();
        write!(f, "return")?;
        for operand in &*operands {
            write!(f, " {}", operand)?;
        }
        let result_types = operation.result_types();
        assert!(!result_types.is_empty(), "Expected result types to be set");
        let result_types = result_types
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, " : {}", result_types)?;
        Ok(())
    }
}

impl Parse for ReturnOp {
    fn op<T: Parse>(parser: &mut Parser<T>, indent: i32) -> Result<Arc<dyn Op>> {
        let _operation_name = parser.expect(TokenKind::BareIdentifier)?;
        let mut operation = Operation::default();
        operation.set_operands(Arc::new(parser.arguments()?));
        let _colon = parser.expect(TokenKind::Colon)?;
        let return_type = parser.expect(TokenKind::IntType)?;
        let return_type = Type::new(return_type.lexeme.clone());
        operation.set_result_types(vec![return_type]);
        operation.set_indent(indent);
        let operation = Arc::new(RwLock::new(operation));
        let op = ReturnOp { operation };
        Ok(Arc::new(op))
    }
}
