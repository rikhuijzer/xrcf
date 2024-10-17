use crate::dialect::arith::set_defining_op;
use crate::dialect::func;
use crate::dialect::llvmir::attribute::LinkageAttr;
use crate::ir::operation;
use crate::ir::operation::OperationName;
use crate::ir::value::Value;
use crate::ir::AnyAttr;
use crate::ir::Block;
use crate::ir::IntegerAttr;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::StrAttr;
use crate::parser::Parser;
use crate::parser::TokenKind;
use crate::Attribute;
use crate::Parse;
use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

pub struct GlobalOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for GlobalOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.mlir.global".to_string())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        if operation.read().unwrap().name() != Self::operation_name() {
            return Err(anyhow::anyhow!(
                "Expected global, got {}",
                operation.read().unwrap().name()
            ));
        }
        Ok(Self { operation })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} ", Self::operation_name().name())?;
        let operation = self.operation().read().unwrap();
        let attributes = operation.attributes().map();
        let attributes = attributes.read().unwrap();
        if let Some(attribute) = attributes.get("linkage") {
            write!(f, "{} ", attribute)?;
        }
        if let Some(attribute) = attributes.get("symbol_name") {
            write!(f, "{}(", attribute)?;
        }
        if let Some(attribute) = attributes.get("value") {
            write!(f, "{}", attribute)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl Parse for GlobalOp {
    fn op<T: Parse>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let _operation_name = parser.advance();
        let attributes = operation::Attributes::new();
        if parser.check(TokenKind::BareIdentifier) {
            if let Some(attribute) = LinkageAttr::parse(parser, "linkage") {
                attributes
                    .map()
                    .write()
                    .unwrap()
                    .insert("linkage".to_string(), Arc::new(attribute));
            }
        }
        let symbol_name = parser.peek();
        if symbol_name.kind != TokenKind::AtIdentifier {
            return Err(anyhow::anyhow!(
                "Expected @identifier, got {:?}",
                symbol_name
            ));
        }
        if let Some(attribute) = StrAttr::parse(parser, "symbol_name") {
            attributes
                .map()
                .write()
                .unwrap()
                .insert("symbol_name".to_string(), Arc::new(attribute));
        }
        if parser.check(TokenKind::LParen) {
            parser.advance();
            if let Some(attribute) = AnyAttr::parse(parser, "value") {
                attributes
                    .map()
                    .write()
                    .unwrap()
                    .insert("value".to_string(), Arc::new(attribute));
            }
        }
        let mut operation = Operation::default();
        operation.set_name(GlobalOp::operation_name());
        operation.set_attributes(attributes);
        operation.set_parent(parent);
        let operation = Arc::new(RwLock::new(operation));
        let op = GlobalOp::from_operation(operation);
        Ok(Arc::new(RwLock::new(op.unwrap())))
    }
}

pub struct FuncOp {
    identifier: String,
    operation: Arc<RwLock<Operation>>,
}

impl FuncOp {
    pub fn identifier(&self) -> &str {
        &self.identifier
    }
    pub fn set_identifier(&mut self, identifier: String) {
        self.identifier = identifier;
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.func".to_string())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(FuncOp {
            identifier: "didn't set identifier for llvm func".to_string(),
            operation,
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
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let identifier = self.identifier.clone();
        func::FuncOp::display_func(self, identifier, f, indent)
    }
}

impl Parse for FuncOp {
    fn op<T: Parse>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let expected_name = FuncOp::operation_name();
        let (op, identifier) = Parser::<T>::parse_func::<FuncOp>(parser, parent, expected_name)?;
        op.write().unwrap().set_identifier(identifier);
        Ok(op)
    }
}

pub struct ConstantOp {
    operation: Arc<RwLock<Operation>>,
}

impl ConstantOp {
    pub fn value(&self) -> Arc<dyn Attribute> {
        let operation = self.operation().try_read().unwrap();
        let attributes = operation.attributes().map();
        let attributes = attributes.try_read().unwrap();
        attributes.get("value").unwrap().clone()
    }
}

impl Op for ConstantOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.mlir.constant".to_string())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(ConstantOp { operation })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().try_read().unwrap();
        let results = operation.results();
        let results = results.try_read().unwrap();
        let result = results.get(0).unwrap();
        let result = result.try_read().unwrap();
        write!(f, "{} = {}", result, Self::operation_name())?;
        write!(f, "(")?;
        let value = self.value();
        write!(f, "{}", value)?;
        write!(f, ")")?;

        let value = value.as_any().downcast_ref::<IntegerAttr>().unwrap();
        write!(f, " : {}", value.typ())?;
        write!(f, "\n")?;

        Ok(())
    }
}

impl Parse for ConstantOp {
    fn op<T: Parse>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        let results = parser.results()?;
        operation.set_results(results.clone());

        let operation_name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(operation_name.lexeme == ConstantOp::operation_name().to_string());
        operation.set_name(ConstantOp::operation_name());
        operation.set_parent(parent);

        let _lparen = parser.expect(TokenKind::LParen)?;
        let integer = Parser::<T>::integer(parser)?;
        let _rparen = parser.expect(TokenKind::RParen)?;

        let attributes = operation.attributes();
        attributes.insert("value", integer);
        let operation = Arc::new(RwLock::new(operation));
        let op = ConstantOp {
            operation: operation.clone(),
        };
        let op = Arc::new(RwLock::new(op));
        set_defining_op(results, op.clone());

        let _colon = parser.expect(TokenKind::Colon)?;
        let _result_type = parser.expect(TokenKind::IntType)?;

        Ok(op)
    }
}

pub struct ReturnOp {
    operation: Arc<RwLock<Operation>>,
}

impl ReturnOp {
    pub fn value(&self) -> Arc<RwLock<Value>> {
        let operand = self.operation.try_read().unwrap().operand(0);
        let operand = operand.try_read().unwrap();
        operand.value()
    }
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.return".to_string())
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
        func::ReturnOp::display_return(self, &name, f, _indent)
    }
}

impl Parse for ReturnOp {
    fn op<T: Parse>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let expected_name = ReturnOp::operation_name();
        let operation = Parser::<T>::return_op(parser, parent, expected_name)?;
        let op = ReturnOp { operation };
        Ok(Arc::new(RwLock::new(op)))
    }
}
