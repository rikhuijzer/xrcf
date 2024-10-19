use crate::ir::Block;
use crate::ir::BlockArgument;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Type;
use crate::ir::Types;
use crate::ir::Value;
use crate::ir::Values;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use crate::Parse;
use crate::Parser;
use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

pub trait Func: Op {
    fn identifier(&self) -> &str;
    fn set_identifier(&mut self, identifier: String);
    fn arguments(&self) -> Result<Values> {
        let operation = self.operation();
        let operation = operation.read().unwrap();
        let arguments = operation.arguments();
        Ok(arguments.clone())
    }
    fn return_types(&self) -> Types {
        let operation = self.operation();
        let operation = operation.read().unwrap();
        let return_types = operation.result_types();
        return_types.clone()
    }
    fn return_type(&self) -> Result<Type> {
        let return_types = self.return_types();
        let return_types = return_types.read().unwrap();
        assert!(!return_types.is_empty(), "Expected result types to be set");
        assert!(return_types.len() == 1, "Expected single result type");
        let typ = return_types[0].read().unwrap().clone();
        Ok(typ)
    }
}

/// Note that the operands of the function are internally
/// represented by `BlockArgument`s, but the textual form is inline.
pub struct FuncOp {
    identifier: String,
    operation: Arc<RwLock<Operation>>,
}

impl Func for FuncOp {
    fn identifier(&self) -> &str {
        &self.identifier
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = identifier;
    }
}

impl FuncOp {
    pub fn display_func(
        op: &dyn Op,
        identifier: String,
        f: &mut Formatter<'_>,
        indent: i32,
    ) -> std::fmt::Result {
        let name = op.operation().try_read().unwrap().name();
        write!(f, "{name} {identifier}(")?;
        let arguments = op
            .operation()
            .try_read()
            .unwrap()
            .arguments()
            .try_read()
            .unwrap()
            .iter()
            .map(|o| o.try_read().unwrap().to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{}", arguments)?;
        write!(f, ")")?;
        let operation = op.operation();
        let result_types = operation.try_read().unwrap().result_types();
        let result_types = result_types.try_read().unwrap();
        if !result_types.is_empty() {
            if result_types.len() == 1 {
                write!(
                    f,
                    " -> {}",
                    result_types.get(0).unwrap().try_read().unwrap()
                )?;
            } else {
                write!(
                    f,
                    " -> ({})",
                    result_types
                        .iter()
                        .map(|t| t.try_read().unwrap().to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                )?;
            }
        }
        let region = op.operation().try_read().unwrap().region();
        if let Some(region) = region {
            let region = region.try_read().unwrap();
            region.display(f, indent)?;
        }
        Ok(())
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("func.func")
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(FuncOp {
            identifier: "didnt set identifier".to_string(),
            operation: operation,
        })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_func(&self) -> bool {
        true
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
        let identifier = self.identifier.clone();
        FuncOp::display_func(self, identifier, f, indent)
    }
}

impl<T: ParserDispatch> Parser<T> {
    fn identifier(&mut self, kind: TokenKind) -> Result<String> {
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
    fn function_argument(&mut self) -> Result<Arc<RwLock<Value>>> {
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
    fn function_arguments(&mut self) -> Result<Values> {
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
    fn result_types(&mut self) -> Result<Types> {
        let mut result_types = vec![];
        if !self.check(TokenKind::Arrow) {
            return Ok(Arc::new(RwLock::new(vec![])));
        } else {
            let _arrow = self.advance();
            while self.check(TokenKind::IntType) {
                let typ = self.advance();
                let typ = Type::new(typ.lexeme.clone());
                let typ = Arc::new(RwLock::new(typ));
                result_types.push(typ);
            }
        }
        Ok(Arc::new(RwLock::new(result_types)))
    }
    pub fn parse_func<O: Op + 'static>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
        expected_name: OperationName,
    ) -> Result<(Arc<RwLock<O>>, String)> {
        // Similar to `FuncOp::parse` in MLIR's `FuncOps.cpp`.
        let result = if parser.peek_n(1).kind == TokenKind::Equal {
            todo!("This case does not occur?");
        } else {
            None
        };
        let operation_name = parser.advance();
        assert!(operation_name.lexeme == expected_name.to_string());
        let identifier = parser.identifier(TokenKind::AtIdentifier).unwrap();
        let mut operation = Operation::default();
        operation.set_name(expected_name.clone());
        operation.set_arguments(parser.function_arguments()?);
        operation.set_result_types(parser.result_types()?);
        operation.set_parent(parent);
        if let Some(result) = result {
            let result = Arc::new(RwLock::new(result));
            let results = Arc::new(RwLock::new(vec![result]));
            operation.set_results(results);
        }
        let operation = Arc::new(RwLock::new(operation));
        let op = O::from_operation_without_verify(operation.clone(), expected_name)?;
        let op = Arc::new(RwLock::new(op));
        let region = parser.region(op.clone())?;
        let mut operation = operation.write().unwrap();
        operation.set_region(Some(region.clone()));

        let mut region = region.write().unwrap();
        region.set_parent(Some(op.clone()));

        Ok((op, identifier))
    }
}

impl Parse for FuncOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let expected_name = FuncOp::operation_name();
        let (op, identifier) = Parser::<T>::parse_func::<FuncOp>(parser, parent, expected_name)?;
        op.write().unwrap().set_identifier(identifier);
        Ok(op)
    }
}

pub struct ReturnOp {
    operation: Arc<RwLock<Operation>>,
}

impl ReturnOp {
    pub fn display_return(
        op: &dyn Op,
        name: &str,
        f: &mut Formatter<'_>,
        _indent: i32,
    ) -> std::fmt::Result {
        let operation = op.operation();
        let operation = operation.read().unwrap();
        write!(f, "{name}")?;
        let operands = operation.operands().clone();
        let operands = operands.read().unwrap();
        for operand in operands.iter() {
            write!(f, " {}", operand.read().unwrap())?;
        }
        let result_types = operation.result_types();
        let result_types = result_types.read().unwrap();
        assert!(!result_types.is_empty(), "Expected result types to be set");
        let result_types = result_types
            .iter()
            .map(|t| t.read().unwrap().to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, " : {}", result_types)?;
        Ok(())
    }
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("return")
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(ReturnOp { operation })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let name = Self::operation_name().to_string();
        ReturnOp::display_return(self, &name, f, _indent)
    }
}

impl<T: ParserDispatch> Parser<T> {
    pub fn return_op<O: Op>(
        &mut self,
        parent: Option<Arc<RwLock<Block>>>,
        expected_name: OperationName,
    ) -> Result<Arc<RwLock<O>>> {
        tracing::debug!("Parsing return op");
        let operation_name = self.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == expected_name.to_string());
        let mut operation = Operation::default();
        assert!(parent.is_some());
        operation.set_name(expected_name.clone());
        operation.set_parent(parent.clone());
        operation.set_operands(self.operands(parent.clone().unwrap())?);
        let _colon = self.expect(TokenKind::Colon)?;
        let return_type = self.expect(TokenKind::IntType)?;
        let return_type = Type::new(return_type.lexeme.clone());
        let result_type = Arc::new(RwLock::new(return_type));
        let result_types = Arc::new(RwLock::new(vec![result_type]));
        operation.set_result_types(result_types);
        let operation = Arc::new(RwLock::new(operation));
        let op = O::from_operation_without_verify(operation.clone(), expected_name)?;
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
}

impl Parse for ReturnOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let expected_name = ReturnOp::operation_name();
        let op = Parser::<T>::return_op::<ReturnOp>(parser, parent, expected_name)?;
        Ok(op)
    }
}