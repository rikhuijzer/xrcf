use crate::dialect::llvmir::attribute::LinkageAttr;
use crate::ir::operation::OperationName;
use crate::ir::Op;
use crate::ir::Operation;
use crate::parser::Parser;
use crate::ir::AnyAttr;
use crate::parser::TokenKind;
use crate::Attribute;
use crate::Parse;
use anyhow::Result;
use std::pin::Pin;
use std::sync::Arc;
use crate::ir::StrAttr;
use std::fmt::Formatter;
use std::fmt::Display;

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
    fn operation(&self) -> Pin<Box<Operation>> {
        self.operation.clone()
    }
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "foobar");
        Ok(())
    }
}

impl Display for GlobalOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

impl Parse for GlobalOp {
    fn operation<T: Parse>(parser: &mut Parser<T>) -> Result<Operation> {
        let name = OperationName::new(GlobalOp::name().to_string());
        let mut attributes: Vec<Arc<dyn Attribute>> = vec![];
        if parser.check(TokenKind::BareIdentifier) {
            if let Some(attribute) = LinkageAttr::parse(parser, "linkage") {
                attributes.push(Arc::new(attribute));
            }
        }
        let symbol_name = parser.peek();
        if symbol_name.kind != TokenKind::AtIdentifier {
            return Err(anyhow::anyhow!("Expected @identifier, got {:?}", symbol_name));
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
        let regions = vec![];
        let parent_block = None;
        let operation = Operation::new(name, attributes, regions, parent_block);
        Ok(operation)
    }
}