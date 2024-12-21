use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
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
use xrcf::parser::Parse;
use xrcf::parser::Parser;
use xrcf::parser::ParserDispatch;
use xrcf::parser::TokenKind;
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

/// Tokenize an ArnoldC operation name.
fn tokenize_arnoldc_name(name: &str) -> Vec<String> {
    let name = name.replace('\'', " \' ");
    name.split_whitespace().map(|s| s.to_string()).collect()
}

impl<T: ParserDispatch> ArnoldParse for Parser<T> {
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

/// `IT'S SHOWTIME`
///
/// We do not immediately parse this into some `arnold::FuncOp` since the
/// rewrite should not only change the op but also return a zero for the status
/// code.
///
/// ```arnoldc
/// ITS SHOWTIME {
///   TALK TO THE HAND "Hello, world!"
/// }
/// ```
/// will be rewritten to
/// ```mlir
/// func.func @main() -> i32 {
///   TALK TO THE HAND "Hello, world!"
///   %0 = arith.constant 0 : i32
///   return %0 : i32
/// }
/// ```
pub struct BeginMainOp {
    operation: Shared<Operation>,
}

impl Op for BeginMainOp {
    fn operation_name() -> OperationName {
        OperationName::new("IT'S SHOWTIME".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        BeginMainOp { operation }
    }
    fn is_func(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let operation = self.operation.rd();
        write!(f, "{} ", operation.name())?;
        let region = operation.region().unwrap();
        write!(f, "()")?;
        region.rd().display(f, indent)?;
        Ok(())
    }
}

impl Parse for BeginMainOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());

        let name = BeginMainOp::operation_name();
        parser.parse_arnold_operation_name_into(name, &mut operation)?;

        let operation = Shared::new(operation.into());
        let op = BeginMainOp {
            operation: operation.clone(),
        };
        let op = Shared::new(op.into());
        let region = parser.parse_region(op.clone())?;
        let mut operation = operation.wr();
        operation.set_region(Some(region.clone()));
        Ok(op)
    }
}

pub struct CallOp {
    operation: Shared<Operation>,
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
    fn new(operation: Shared<Operation>) -> Self {
        CallOp {
            operation,
            identifier: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}()", self.identifier().unwrap())
    }
}

impl Parse for CallOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let identifier = parser.expect(TokenKind::BareIdentifier)?;
        let identifier = identifier.lexeme.clone();
        parser.expect(TokenKind::LParen)?;
        parser.expect(TokenKind::RParen)?;
        let operation = Shared::new(operation.into());
        let op = CallOp {
            operation: operation.clone(),
            identifier: Some(identifier),
        };
        Ok(Shared::new(op.into()))
    }
}

pub struct DeclareIntOp {
    operation: Shared<Operation>,
}

impl Op for DeclareIntOp {
    fn operation_name() -> OperationName {
        OperationName::new("HEY CHRISTMAS TREE".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        DeclareIntOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", Self::operation_name())?;
        write!(f, " {}", self.operation().rd().results())?;
        Ok(())
    }
}

impl Parse for DeclareIntOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let name = DeclareIntOp::operation_name();
        parser.parse_arnold_operation_name_into(name, &mut operation)?;
        let result = parser.parse_op_result_into(TOKEN_KIND, &mut operation)?;
        let operation = Shared::new(operation.into());
        let op = DeclareIntOp { operation };
        let op = Shared::new(op.into());
        result.set_defining_op(Some(op.clone()));
        let typ = IntegerType::new(16);
        let typ = Shared::new(typ.into());
        result.set_typ(typ);
        Ok(op)
    }
}

/// `BECAUSE I'M GOING TO SAY PLEASE`
///
/// Examples:
/// ```arnoldc
/// BECAUSE I'M GOING TO SAY PLEASE x
///   TALK TO THE HAND "x was true"
/// BULLSHIT
///   TALK TO THE HAND "x was false"
/// YOU HAVE NO RESPECT FOR LOGIC
/// ```
pub struct IfOp {
    operation: Shared<Operation>,
    then: Option<Shared<Region>>,
    els: Option<Shared<Region>>,
}

impl IfOp {
    pub fn then(&self) -> Option<Shared<Region>> {
        self.then.clone()
    }
    pub fn els(&self) -> Option<Shared<Region>> {
        self.els.clone()
    }
}

impl Op for IfOp {
    fn operation_name() -> OperationName {
        OperationName::new("BECAUSE I'M GOING TO SAY PLEASE".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        IfOp {
            operation,
            then: None,
            els: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
}

impl Parse for IfOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let name = IfOp::operation_name();
        parser.parse_arnold_operation_name_into(name, &mut operation)?;
        parser.parse_op_operand_into(parent.clone().unwrap(), TOKEN_KIND, &mut operation)?;
        let operation = Shared::new(operation.into());
        let op = IfOp {
            operation: operation.clone(),
            then: None,
            els: None,
        };
        let op = Shared::new(op.into());
        let then = parser.parse_region(op.clone())?;
        let else_keyword = parser.expect(TokenKind::BareIdentifier)?;
        if else_keyword.lexeme != "BULLSHIT" {
            panic!("Expected BULLSHIT but got {}", else_keyword.lexeme);
        }
        let els = parser.parse_region(op.clone())?;
        let op_write = op.clone();
        let mut op_write = op_write.wr();
        op_write.then = Some(then);
        op_write.els = Some(els);
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
    operation: Shared<Operation>,
}

impl PrintOp {
    const TEXT_INDEX: usize = 0;
    pub fn text(&self) -> Shared<OpOperand> {
        self.operation
            .rd()
            .operand(Self::TEXT_INDEX)
            .expect("Operand not set")
    }
    pub fn set_text(&mut self, text: Shared<OpOperand>) {
        self.operation.wr().set_operand(Self::TEXT_INDEX, text);
    }
}

impl Op for PrintOp {
    fn operation_name() -> OperationName {
        OperationName::new("TALK TO THE HAND".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        PrintOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", Self::operation_name())?;
        write!(f, " {}", self.text().rd())?;
        Ok(())
    }
}

impl Parse for PrintOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let name = PrintOp::operation_name();
        parser.parse_arnold_operation_name_into(name, &mut operation)?;
        let operation = Shared::new(operation.into());
        let text = parser.parse_op_operand(parent.clone().unwrap(), TOKEN_KIND)?;
        let mut op = PrintOp {
            operation: operation.clone(),
        };
        op.set_text(text);
        Ok(Shared::new(op.into()))
    }
}

pub struct SetInitialValueOp {
    operation: Shared<Operation>,
}

impl SetInitialValueOp {
    pub fn value(&self) -> Arc<dyn Attribute> {
        self.operation.rd().attributes().get("value").unwrap()
    }
}

impl Op for SetInitialValueOp {
    fn operation_name() -> OperationName {
        OperationName::new("YOU SET US UP".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        SetInitialValueOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
}

impl Parse for SetInitialValueOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let name = SetInitialValueOp::operation_name();
        parser.parse_arnold_operation_name_into(name, &mut operation)?;

        let next = parser.peek();
        if next.lexeme.starts_with('@') {
            parser.parse_arnold_constant_into(&mut operation)?;
        }

        let operation = Shared::new(operation.into());
        let op = SetInitialValueOp {
            operation: operation.clone(),
        };
        Ok(Shared::new(op.into()))
    }
}
