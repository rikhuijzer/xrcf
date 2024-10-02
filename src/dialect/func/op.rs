use crate::ir::Block;
use crate::ir::BlockArgument;
use crate::ir::Op;
use crate::ir::OpResult;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Type;
use crate::ir::Value;
use crate::ir::Values;
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn assignments(&self) -> Result<Values> {
        let operation = self.operation();
        let operation = operation.read().unwrap();
        let arguments = operation.arguments();
        Ok(arguments.clone())
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        write!(f, "func.func {}(", self.identifier)?;
        let joined = self
            .operation()
            .read()
            .unwrap()
            .arguments()
            .read()
            .unwrap()
            .iter()
            .map(|o| o.read().unwrap().to_string())
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
        if let Some(region) = region {
            let region = region.read().unwrap();
            region.display(f, indent)?;
        }
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
    /// %arg0 : i64,
    pub fn function_argument(&mut self) -> Result<Arc<RwLock<Value>>> {
        let identifier = self.expect(TokenKind::PercentIdentifier)?;
        let name = identifier.lexeme.clone();
        let typ = if self.check(TokenKind::Colon) {
            self.advance();
            let typ = self.advance();
            Type::new(typ.lexeme.clone())
        } else {
            Type::new("any".to_string())
        };
        let arg = Value::BlockArgument(BlockArgument::new(name, typ));
        let operand = Arc::new(RwLock::new(arg));
        if self.check(TokenKind::Comma) {
            self.advance();
        }
        Ok(operand)
    }
    /// Parse `(%arg0 : i64, %arg1 : i64)`.
    pub fn function_arguments(&mut self) -> Result<Values> {
        let _lparen = self.expect(TokenKind::LParen)?;
        let mut operands = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            operands.push(self.function_argument()?);
        }
        if self.check(TokenKind::RParen) {
            let _rparen = self.advance();
        } else if self.check(TokenKind::Colon) {
        } else {
            return Err(anyhow::anyhow!("Expected ')', got {:?}", self.peek().kind));
        }
        Ok(Arc::new(RwLock::new(operands)))
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
    fn op<T: Parse>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        // Similar to `FuncOp::parse` in MLIR's `FuncOps.cpp`.
        let result = if parser.peek_n(1).kind == TokenKind::Equal {
            todo!("This case does not occur?");
        } else {
            None
        };
        let operation_name = parser.advance();
        assert!(operation_name.lexeme == "func.func");
        let identifier = parser.identifier(TokenKind::AtIdentifier).unwrap();
        let mut operation = Operation::default();
        operation.set_name(FuncOp::operation_name());
        operation.set_arguments(parser.function_arguments()?);
        operation.set_result_types(parser.result_types()?);
        operation.set_parent(parent);
        if let Some(result) = result {
            let result = Arc::new(RwLock::new(result));
            let results = Arc::new(RwLock::new(vec![result]));
            operation.set_results(results);
        }
        let operation = Arc::new(RwLock::new(operation));
        let op = FuncOp {
            identifier,
            operation: operation.clone(),
        };
        let op = Arc::new(RwLock::new(op));
        let region = parser.region(op.clone())?;
        let mut operation = operation.write().unwrap();
        operation.set_region(Some(region.clone()));

        let mut region = region.write().unwrap();
        region.set_parent(Some(op.clone()));

        Ok(op)
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation();
        let operation = operation.read().unwrap();
        write!(f, "return")?;
        let operands = operation.operands().clone();
        let operands = operands.read().unwrap();
        for operand in operands.iter() {
            write!(f, " {}", operand.read().unwrap())?;
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
    fn op<T: Parse>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == "return");
        let mut operation = Operation::default();
        assert!(parent.is_some());
        operation.set_parent(parent.clone());
        operation.set_operands(parser.operands(parent.clone().unwrap())?);
        let _colon = parser.expect(TokenKind::Colon)?;
        let return_type = parser.expect(TokenKind::IntType)?;
        let return_type = Type::new(return_type.lexeme.clone());
        operation.set_result_types(vec![return_type]);
        let operation = Arc::new(RwLock::new(operation));
        let op = ReturnOp { operation };
        Ok(Arc::new(RwLock::new(op)))
    }
}
