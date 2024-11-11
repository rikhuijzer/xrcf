use crate::ir::OpOperand;
use crate::ir::OpOperands;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

pub trait Type {
    /// Display the type.
    ///
    /// This has to be implemented by each type so that calls to `Display::fmt`
    /// on a `dyn Type` can be delegated to the type's `display` method.
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
    fn as_any(&self) -> &dyn std::any::Any;
}

impl Display for dyn Type {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

/// Interface to parse a type.
///
/// This trait can be implemented by a dialect to parse types from a string.
pub trait TypeParse {
    fn parse_type(src: &str) -> Result<Arc<RwLock<dyn Type>>>;
}

/// Interface to convert a type from one dialect to another.
///
/// This trait can be implemented by a dialect to lower types to their
/// corresponding dialect types.
pub trait TypeConvert {
    fn convert_str(src: &str) -> Result<Arc<RwLock<dyn Type>>>;
    /// Convert a `Type` from one dialect to another.
    ///
    /// This method can be reimplemented to compare types directly instead of
    /// converting to a string first.
    fn convert_type(from: &Arc<RwLock<dyn Type>>) -> Result<Arc<RwLock<dyn Type>>> {
        let from = from.try_read().unwrap();
        let typ = Self::convert_str(&from.to_string())?;
        Ok(typ)
    }
}

pub struct AnyType {
    typ: String,
}

impl AnyType {
    pub fn new(typ: &str) -> Self {
        Self {
            typ: typ.to_string(),
        }
    }
    pub fn typ(&self) -> String {
        self.typ.clone()
    }
    pub fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.typ)
    }
}

impl Type for AnyType {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Represent an integer type such as i32 or i64.
///
/// Just like in LLVM, this does not include the sign bit since the sign does
/// not matter for 2s complement integer arithmetic.
#[derive(Debug, Clone, Copy)]
pub struct IntegerType {
    num_bits: u64,
}

impl IntegerType {
    pub fn new(num_bits: u64) -> Self {
        Self { num_bits }
    }
    pub fn from_str(s: &str) -> Self {
        let s = s.strip_prefix("i").expect("no i prefix");
        let num_bits = s.parse::<u64>().unwrap();
        Self { num_bits }
    }
}

impl Type for IntegerType {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "i{}", self.num_bits)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Display for IntegerType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StringType;

impl StringType {
    pub fn new() -> Self {
        Self
    }
}

impl Type for StringType {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "str")
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Arbitrary precision integer.
pub struct APInt {
    #[allow(dead_code)]
    num_bits: u64,
    value: u64,
    is_signed: bool,
}

impl APInt {
    pub fn new(num_bits: u64, value: u64, is_signed: bool) -> Self {
        Self {
            num_bits,
            value,
            is_signed,
        }
    }
    pub fn from_str(typ: &str, value: &str) -> Self {
        let typ = IntegerType::from_str(typ);
        let value = value.parse::<u64>().unwrap();
        Self::new(typ.num_bits, value, true)
    }
    pub fn value(&self) -> u64 {
        self.value
    }
    pub fn is_signed(&self) -> bool {
        self.is_signed
    }
}

impl Type for APInt {
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Display for APInt {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

/// A collection of `Type`s.
///
/// Provides some convenience methods around [Type]s.
#[derive(Clone)]
pub struct Types {
    types: Vec<Arc<RwLock<dyn Type>>>,
}

impl Types {
    pub fn from_vec(types: Vec<Arc<RwLock<dyn Type>>>) -> Self {
        Self { types }
    }
    pub fn vec(&self) -> Vec<Arc<RwLock<dyn Type>>> {
        self.types.clone()
    }
}

impl Default for Types {
    fn default() -> Self {
        Self { types: vec![] }
    }
}

impl Display for Types {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let joined = self
            .types
            .iter()
            .map(|t| t.try_read().unwrap().to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{}", joined)
    }
}

impl<T: ParserDispatch> Parser<T> {
    /// Verify that the type of an operand matches a given type.
    ///
    /// Useful during the parsing of certain ops where the operand type is
    /// expected to match a given type.
    pub fn verify_type(
        &mut self,
        operand: Arc<RwLock<OpOperand>>,
        typ: Arc<RwLock<dyn Type>>,
    ) -> Result<()> {
        let operand = operand.try_read().unwrap();
        let operand_typ = operand.typ()?;
        let operand_typ = operand_typ.try_read().unwrap();
        let token = self.previous().clone();
        let typ = typ.try_read().unwrap();
        if operand_typ.to_string() != typ.to_string() {
            let msg = format!(
                "Expected {} due to {}, but got {}",
                operand_typ, operand, typ
            );
            let msg = self.error(&token, &msg);
            return Err(anyhow::anyhow!(msg));
        }
        Ok(())
    }
    pub fn parse_type(&mut self) -> Result<Arc<RwLock<dyn Type>>> {
        T::parse_type(self)
    }
    /// Parse types until a closing parenthesis.
    pub fn parse_types(&mut self) -> Result<Vec<Arc<RwLock<dyn Type>>>> {
        let mut types = vec![];
        while !self.check(TokenKind::RParen) {
            let typ = self.parse_type()?;
            types.push(typ);
            if self.check(TokenKind::Comma) {
                self.advance();
            }
        }
        Ok(types)
    }
    /// Parse types and verify that they match the given operands.
    ///
    /// For example, can be used to verify that `%0` has type `i32` in:
    ///
    /// ```mlir
    /// %0 = arith.constant 42 : i32
    /// llvm.call @printf(%0) : (i32) -> (i32)
    /// ```
    pub fn parse_types_for_op_operands(&mut self, operands: OpOperands) -> Result<()> {
        let types = self.parse_types()?;
        if types.len() != operands.vec().try_read().unwrap().len() {
            let msg = format!(
                "Expected {} types but got {}",
                operands.vec().try_read().unwrap().len(),
                types.len()
            );
            return Err(anyhow::anyhow!(msg));
        }
        let operands = operands.vec();
        let operands = operands.try_read().unwrap();
        for x in operands.iter().zip(types.iter()) {
            let (operand, typ) = x;
            self.verify_type(operand.clone(), typ.clone())?;
        }
        Ok(())
    }
}
