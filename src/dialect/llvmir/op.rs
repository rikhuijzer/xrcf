use crate::dialect::llvmir::attribute::LinkageAttr;
use crate::ir::operation::OperationName;
use crate::ir::AnyAttr;
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
    fn from_operation(operation: Arc<RwLock<Operation>>) -> Result<Self> {
        if operation.read().unwrap().name() != Self::operation_name().name() {
            return Err(anyhow::anyhow!(
                "Expected global, got {}",
                operation.read().unwrap().name()
            ));
        }
        Ok(Self { operation })
    }
    fn set_indentation(&self, indentation: i32) {
        let mut operation = self.operation.write().unwrap();
        operation.set_indentation(indentation);
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ", Self::operation_name().name())?;
        let operation = self.operation().read().unwrap();
        for attribute in operation.attributes() {
            write!(f, "{}", attribute)?;
            if attribute.name() == "symbol_name" {
                write!(f, "(")?;
            } else if attribute.name() == "value" {
            } else {
                write!(f, " ")?;
            }
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl Parse for GlobalOp {
    fn op<T: Parse>(parser: &mut Parser<T>) -> Result<Arc<dyn Op>> {
        let _operation_name = parser.advance();
        let name = GlobalOp::operation_name();
        let mut attributes: Vec<Arc<dyn Attribute>> = vec![];
        if parser.check(TokenKind::BareIdentifier) {
            if let Some(attribute) = LinkageAttr::parse(parser, "linkage") {
                attributes.push(Arc::new(attribute));
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
            attributes.push(Arc::new(attribute));
        }
        if parser.check(TokenKind::LParen) {
            parser.advance();
            if let Some(attribute) = AnyAttr::parse(parser, "value") {
                attributes.push(Arc::new(attribute));
            }
        }
        println!("{:?}", parser.advance());
        let mut operation = Operation::default();
        operation.set_name(name).set_attributes(attributes);
        let operation = Arc::new(RwLock::new(operation));
        let op = GlobalOp::from_operation(operation);
        Ok(Arc::new(op.unwrap()))
    }
}
