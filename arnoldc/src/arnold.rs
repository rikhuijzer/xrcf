use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::dialect::func::Func;
use xrcf::ir::APInt;
use xrcf::ir::Attribute;
use xrcf::ir::Block;
use xrcf::ir::GuardedOperation;
use xrcf::ir::IntegerAttr;
use xrcf::ir::IntegerType;
use xrcf::ir::Op;
use xrcf::ir::OpOperand;
use xrcf::ir::Operation;
use xrcf::ir::OperationName;
use xrcf::parser::Parse;
use xrcf::parser::Parser;
use xrcf::parser::ParserDispatch;
use xrcf::parser::TokenKind;

/// The token kind used for variables in ArnoldC.
///
/// In ArnoldC variables are always bare identifiers meaning `x` is a valid
/// variable. For example, percent identifiers like `%x` are not valid.
const TOKEN_KIND: TokenKind = TokenKind::BareIdentifier;

/// Variables are always 16-bit signed integers in ArnoldC.
fn arnold_integer(value: u64) -> APInt {
    APInt::new(16, value, true)
}

fn arnold_attribute(value: u64) -> IntegerAttr {
    let typ = IntegerType::new(16);
    let value = arnold_integer(value);
    IntegerAttr::new(typ, value)
}

/// Arnold-specific parsing methods.
///
/// This makes the methods available on the `parser` object.
///
/// Example:
/// ```rust
/// parser.parse_arnold_op_operand(parent)
/// ```
trait ArnoldParse {
    fn parse_arnold_constant_into(&mut self, operation: &mut Operation) -> Result<()>;
    fn parse_arnold_operation_name_into(
        &mut self,
        name: OperationName,
        operation: &mut Operation,
    ) -> Result<()>;
}

impl<T: ParserDispatch> ArnoldParse for Parser<T> {
    /// Parse a constant like `@NO PROBLEMO` into the [Operation].
    fn parse_arnold_constant_into(&mut self, operation: &mut Operation) -> Result<()> {
        let next = self.expect(TokenKind::AtIdentifier)?;
        assert!(next.lexeme.starts_with('@'));
        let next_next = self.advance();
        let constant = format!("{} {}", next.lexeme, next_next.lexeme);
        let constant = if constant == "@NO PROBLEMO" {
            arnold_attribute(0)
        } else if constant == "@I LIED" {
            arnold_attribute(1)
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
        let text = name.to_string();
        let parts = text.split_whitespace().collect::<Vec<&str>>();
        for part in parts {
            let next = self.peek();
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

pub struct CallOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
}

impl CallOp {
    pub fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
}

impl Op for CallOp {
    fn operation_name() -> OperationName {
        OperationName::new("call".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        CallOp {
            operation,
            identifier: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}()", self.identifier().unwrap())
    }
}

impl Parse for CallOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let identifier = parser.expect(TokenKind::BareIdentifier)?;
        let identifier = identifier.lexeme.clone();
        parser.expect(TokenKind::LParen)?;
        parser.expect(TokenKind::RParen)?;
        let operation = Arc::new(RwLock::new(operation));
        let op = CallOp {
            operation: operation.clone(),
            identifier: Some(identifier),
        };
        Ok(Arc::new(RwLock::new(op)))
    }
}

pub struct DeclareIntOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for DeclareIntOp {
    fn operation_name() -> OperationName {
        OperationName::new("HEY CHRISTMAS TREE".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        DeclareIntOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", Self::operation_name())?;
        write!(f, " {}", self.operation().read().unwrap().results())?;
        Ok(())
    }
}

impl Parse for DeclareIntOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let name = DeclareIntOp::operation_name();
        parser.parse_arnold_operation_name_into(name, &mut operation)?;
        parser.parse_op_results_into(TOKEN_KIND, &mut operation)?;
        let operation = Arc::new(RwLock::new(operation));
        let op = DeclareIntOp { operation };
        Ok(Arc::new(RwLock::new(op)))
    }
}

pub struct FuncOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
}

impl Func for FuncOp {
    fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("def".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        FuncOp {
            operation,
            identifier: None,
        }
    }
    fn is_func(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let operation = self.operation.try_read().unwrap();
        write!(f, "{} ", operation.name())?;
        write!(f, "{}", self.identifier().unwrap())?;
        let region = operation.region().unwrap();
        let region = region.try_read().unwrap();
        write!(f, "()")?;
        region.display(f, indent)?;
        Ok(())
    }
}

impl Parse for FuncOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());

        parser.parse_operation_name_into::<FuncOp>(&mut operation)?;
        let identifier = parser.expect(TokenKind::BareIdentifier)?;
        let identifier = identifier.lexeme.clone();

        let operation = Arc::new(RwLock::new(operation));
        let op = FuncOp {
            operation: operation.clone(),
            identifier: Some(identifier),
        };
        let op = Arc::new(RwLock::new(op));
        parser.expect(TokenKind::LParen)?;
        parser.expect(TokenKind::RParen)?;
        let region = parser.region(op.clone())?;
        let mut operation = operation.write().unwrap();
        operation.set_region(Some(region.clone()));
        Ok(op)
    }
}

/// `TALK TO THE HAND`
///
/// Examples:
/// ```arnoldc
/// TALK TO THE HAND "Hello, world!"
///
/// TALK TO THE HAND x
/// ```
pub struct PrintOp {
    operation: Arc<RwLock<Operation>>,
}

impl PrintOp {
    const TEXT_INDEX: usize = 0;
    pub fn text(&self) -> Arc<RwLock<OpOperand>> {
        self.operation
            .operand(Self::TEXT_INDEX)
            .expect("Operand not set")
    }
    pub fn set_text(&mut self, text: Arc<RwLock<OpOperand>>) {
        self.operation.set_operand(Self::TEXT_INDEX, text);
    }
}

impl Op for PrintOp {
    fn operation_name() -> OperationName {
        OperationName::new("TALK TO THE HAND".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        PrintOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", Self::operation_name())?;
        write!(f, " {}", self.text().try_read().unwrap())?;
        Ok(())
    }
}

impl Parse for PrintOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let name = PrintOp::operation_name();
        parser.parse_arnold_operation_name_into(name, &mut operation)?;
        let operation = Arc::new(RwLock::new(operation));
        let text = parser.parse_op_operand(parent.clone().unwrap(), TOKEN_KIND)?;
        let mut op = PrintOp {
            operation: operation.clone(),
        };
        op.set_text(text);
        Ok(Arc::new(RwLock::new(op)))
    }
}

pub struct SetInitialValueOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for SetInitialValueOp {
    fn operation_name() -> OperationName {
        OperationName::new("YOU SET US UP".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        SetInitialValueOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
}

impl Parse for SetInitialValueOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let name = SetInitialValueOp::operation_name();
        parser.parse_arnold_operation_name_into(name, &mut operation)?;

        let next = parser.peek();
        if next.lexeme.starts_with('@') {
            parser.parse_arnold_constant_into(&mut operation)?;
        }

        let operation = Arc::new(RwLock::new(operation));
        let op = SetInitialValueOp {
            operation: operation.clone(),
        };
        Ok(Arc::new(RwLock::new(op)))
    }
}
