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
use std::fmt::Display;
use std::fmt::Formatter;
use std::pin::Pin;
use std::sync::Arc;

pub struct GlobalOp {
    operation: Pin<Box<Operation>>,
}

impl Op for GlobalOp {
    fn name() -> &'static str {
        "llvm.mlir.global"
    }
    fn from_operation(operation: Pin<Box<Operation>>) -> Result<Self> {
        if operation.name() != Self::name() {
            return Err(anyhow::anyhow!("Expected global, got {}", operation.name()));
        }
        Ok(Self {
            operation: operation,
        })
    }
    fn operation(&self) -> &Pin<Box<Operation>> {
        &self.operation
    }
}

impl Display for GlobalOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ", Self::name())?;
        for attribute in self.operation().attributes() {
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
        let name = OperationName::new(GlobalOp::name().to_string());
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
        let op = GlobalOp::from_operation(Box::pin(operation));
        Ok(Arc::new(op.unwrap()))
    }
}
