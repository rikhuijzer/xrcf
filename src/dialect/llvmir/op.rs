use crate::dialect::llvmir::attribute::LinkageAttr;
use crate::ir::operation;
use crate::ir::operation::OperationName;
use crate::ir::AnyAttr;
use crate::ir::Block;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::StrAttr;
use crate::parser::Parser;
use crate::parser::TokenKind;
use crate::Attribute;
use crate::Parse;
use anyhow::Result;
use std::collections::HashMap;
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
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} ", Self::operation_name().name())?;
        let operation = self.operation().read().unwrap();
        for (name, attribute) in operation.attributes().read().unwrap().iter() {
            write!(f, "{}", attribute)?;
            if name == "symbol_name" {
                write!(f, "(")?;
            } else if name == "value" {
            } else {
                write!(f, " ")?;
            }
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
        let name = GlobalOp::operation_name();
        let attributes: operation::Attributes = Arc::new(RwLock::new(HashMap::new()));
        if parser.check(TokenKind::BareIdentifier) {
            if let Some(attribute) = LinkageAttr::parse(parser, "linkage") {
                attributes
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
                .write()
                .unwrap()
                .insert("symbol_name".to_string(), Arc::new(attribute));
        }
        if parser.check(TokenKind::LParen) {
            parser.advance();
            if let Some(attribute) = AnyAttr::parse(parser, "value") {
                attributes
                    .write()
                    .unwrap()
                    .insert("value".to_string(), Arc::new(attribute));
            }
        }
        println!("{:?}", parser.advance());
        let mut operation = Operation::default();
        operation.set_name(name);
        operation.set_attributes(attributes);
        operation.set_parent(parent);
        let operation = Arc::new(RwLock::new(operation));
        let op = GlobalOp::from_operation(operation);
        Ok(Arc::new(RwLock::new(op.unwrap())))
    }
}
