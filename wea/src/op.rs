use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use xrcf::frontend::Parse;
use xrcf::frontend::Parser;
use xrcf::frontend::ParserDispatch;
use xrcf::frontend::TokenKind;
use xrcf::ir::APInt;
use xrcf::ir::Attribute;
use xrcf::ir::Block;
use xrcf::ir::IntegerAttr;
use xrcf::ir::IntegerType;
use xrcf::ir::Op;
use xrcf::ir::OpOperand;
use xrcf::ir::Operation;
use xrcf::ir::OperationName;
use xrcf::ir::Region;
use xrcf::shared::Shared;
use xrcf::shared::SharedExt;

/// The token kind used for variables in ArnoldC.
///
/// In ArnoldC variables are always bare identifiers meaning `x` is a valid
/// variable. For example, percent identifiers like `%x` are not valid.
const TOKEN_KIND: TokenKind = TokenKind::BareIdentifier;

/// Variables are always 16-bit signed integers in ArnoldC.
fn arnold_integer(value: u64) -> APInt {
    APInt::new(16, value, true)
}

fn arnold_attribute(value: u64, num_bits: u64) -> IntegerAttr {
    let typ = IntegerType::new(num_bits);
    let value = arnold_integer(value);
    IntegerAttr::new(typ, value)
}

trait WeaParse {
    fn parse_arnold_constant_into(&mut self, operation: &mut Operation) -> Result<()>;
    fn parse_arnold_operation_name_into(
        &mut self,
        name: OperationName,
        operation: &mut Operation,
    ) -> Result<()>;
}

/// Tokenize an ArnoldC operation name.
fn tokenize_arnoldc_name(name: &str) -> Vec<String> {
    let name = name.replace('\'', " \' ");
    name.split_whitespace().map(|s| s.to_string()).collect()
}

impl<T: ParserDispatch> WeaParse for Parser<T> {
    /// Parse a constant like `@NO PROBLEMO` into the [Operation].
    fn parse_arnold_constant_into(&mut self, operation: &mut Operation) -> Result<()> {
        let next = self.expect(TokenKind::AtIdentifier)?;
        assert!(next.lexeme.starts_with('@'));
        let next_next = self.advance();
        let constant = format!("{} {}", next.lexeme, next_next.lexeme);
        let constant = if constant == "@NO PROBLEMO" {
            arnold_attribute(1, 1)
        } else if constant == "@I LIED" {
            arnold_attribute(0, 1)
        } else {
            return Err(anyhow::anyhow!("Unknown constant: {}", constant));
        };
        let constant: Arc<dyn Attribute> = Arc::new(constant);
        operation.attributes().insert("value", constant);
        Ok(())
    }
    /// Parse an operation name like `TALK TO THE HAND` into the [Operation].
    fn parse_arnold_operation_name_into(
        &mut self,
        name: OperationName,
        operation: &mut Operation,
    ) -> Result<()> {
        let parts = tokenize_arnoldc_name(&name.to_string());
        for part in parts {
            let next = self.peek();
            if next.kind == TokenKind::SingleQuote {
                self.advance();
                continue;
            }
            if next.kind != TokenKind::BareIdentifier {
                return Err(anyhow::anyhow!(
                    "Expected part {} but got {:?}",
                    part,
                    next.kind
                ));
            }
            if next.lexeme != part {
                return Err(anyhow::anyhow!(
                    "Expected part {} but got {}",
                    part,
                    next.lexeme
                ));
            }
            self.advance();
        }
        operation.set_name(name);
        Ok(())
    }
}

pub struct FuncOp {
    operation: Shared<Operation>,
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("fn".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        FuncOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", Self::operation_name())?;
        Ok(())
    }
}

impl Parse for FuncOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let name = FuncOp::operation_name();
        parser.parse_arnold_operation_name_into(name, &mut operation)?;
        let operation = Shared::new(operation.into());
        let op = FuncOp::new(operation.clone());
        Ok(Shared::new(op.into()))
    }
}

/// `+`
pub struct PlusOp {
    operation: Shared<Operation>,
}

impl Op for PlusOp {
    fn operation_name() -> OperationName {
        OperationName::new("+".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        PlusOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", Self::operation_name())?;
        Ok(())
    }
}

impl Parse for PlusOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let name = PlusOp::operation_name();
        parser.parse_arnold_operation_name_into(name, &mut operation)?;
        let operation = Shared::new(operation.into());
        let text = parser.parse_op_operand(parent.clone().unwrap(), TOKEN_KIND)?;
        let mut op = PlusOp {
            operation: operation.clone(),
        };
        Ok(Shared::new(op.into()))
    }
}
