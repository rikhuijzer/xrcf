use crate::frontend::Parser;
use crate::frontend::ParserDispatch;
use crate::frontend::TokenKind;
use crate::ir::bytes_to_llvm_string;
use crate::ir::escape;
use crate::ir::llvm_string_to_bytes;
use crate::ir::unescape;
use crate::ir::APInt;
use crate::ir::IntegerType;
use crate::ir::StringType;
use crate::ir::Type;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use std::collections::HashMap;
use std::fmt::Display;
use std::fmt::Formatter;
use std::str::FromStr;
use std::sync::Arc;

/// Attributes are known-constant values of operations (a variable is not allowed).
/// Attributes belong to operations and can be used to, for example, specify
/// a SSA value.
pub trait Attribute {
    fn from_str(value: &str) -> Self
    where
        Self: Sized;
    fn parse<T: ParserDispatch>(parser: &mut Parser<T>) -> Option<Self>
    where
        Self: Sized;

    fn as_any(&self) -> &dyn std::any::Any;
    fn value(&self) -> String;
    fn typ(&self) -> Shared<dyn Type>;
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

impl Display for dyn Attribute {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

pub struct BooleanAttr {
    value: bool,
}

/// MLIR-style boolean attribute (e.g., `true : i1`).
impl BooleanAttr {
    pub fn new(value: bool) -> Self {
        Self { value }
    }
}

impl Attribute for BooleanAttr {
    fn from_str(value: &str) -> Self {
        let value = match value {
            "true" => true,
            "false" => false,
            _ => panic!("invalid boolean value: {value}"),
        };
        Self { value }
    }
    fn typ(&self) -> Shared<dyn Type> {
        Shared::new(IntegerType::from_str("i1").unwrap().into())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn value(&self) -> String {
        self.value.to_string()
    }
    fn parse<T: ParserDispatch>(parser: &mut Parser<T>) -> Option<Self> {
        Some(Self::from_str(&parser.peek().lexeme))
    }
}

/// Integer attribute (e.g., `42 : i64`).
pub struct IntegerAttr {
    // The type of the integer: specifies the precision.
    typ: IntegerType,
    // An arbitrary precision integer value.
    value: APInt,
}

impl IntegerAttr {
    pub fn new(typ: IntegerType, value: APInt) -> Self {
        Self { typ, value }
    }
    pub fn i64(&self) -> i64 {
        let int = &self.value;
        assert!(int.is_signed(), "must be signed integer");
        int.value() as i64
    }
    pub fn u64(&self) -> u64 {
        assert!(!self.value.is_signed(), "must be unsigned integer");
        self.value.value()
    }
}

impl Display for IntegerAttr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

impl Attribute for IntegerAttr {
    fn from_str(value: &str) -> Self {
        Self {
            typ: IntegerType::new(64),
            value: APInt::new(64, value.parse::<u64>().unwrap(), true),
        }
    }
    fn typ(&self) -> Shared<dyn Type> {
        Shared::new(self.typ.into())
    }
    fn parse<T: ParserDispatch>(_parser: &mut Parser<T>) -> Option<Self> {
        todo!()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn value(&self) -> String {
        self.value.to_string()
    }
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} : {}", self.value, self.typ)
    }
}

impl IntegerAttr {
    pub fn add(a: &IntegerAttr, b: &IntegerAttr) -> IntegerAttr {
        let a_value = a.value().parse::<u64>().unwrap();
        let b_value = b.value().parse::<u64>().unwrap();
        let value = a_value + b_value;
        let value = APInt::new(64, value, true);
        IntegerAttr::new(a.typ, value)
    }
}

/// UTF-8 encoded string.
#[derive(Clone)]
pub struct StringAttr {
    value: Vec<u8>,
}

impl StringAttr {
    pub fn new(value: Vec<u8>) -> Self {
        Self { value }
    }
    pub fn c_string(&self) -> Vec<u8> {
        let mut value = self.value.clone();
        let null_byte = 0;
        if let Some(last) = value.last() {
            if *last != null_byte {
                value.push(null_byte);
            }
        }
        value
    }
}

impl Attribute for StringAttr {
    fn from_str(src: &str) -> Self {
        let text = unescape(src);
        let value = llvm_string_to_bytes(&text);
        Self { value }
    }
    fn typ(&self) -> Shared<dyn Type> {
        Shared::new(StringType::new().into())
    }
    fn parse<T: ParserDispatch>(_parser: &mut Parser<T>) -> Option<Self> {
        todo!()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn value(&self) -> String {
        String::from_utf8(self.value.clone()).unwrap()
    }
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let text = bytes_to_llvm_string(&self.value);
        let text = escape(&text);
        write!(f, "\"{}\"", text)
    }
}

impl Display for StringAttr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f)
    }
}

pub struct AnyAttr {
    value: String,
}

impl Attribute for AnyAttr {
    fn from_str(value: &str) -> Self {
        Self {
            value: value.to_string(),
        }
    }
    fn typ(&self) -> Shared<dyn Type> {
        Shared::new(StringType::new().into())
    }
    fn parse<T: ParserDispatch>(parser: &mut Parser<T>) -> Option<Self> {
        let value = parser.advance();
        Some(Self {
            value: value.lexeme.to_string(),
        })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn value(&self) -> String {
        self.value.clone()
    }
    fn display(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[derive(Clone)]
pub struct Attributes {
    map: Shared<HashMap<String, Arc<dyn Attribute>>>,
}

impl Attributes {
    pub fn new() -> Self {
        Self {
            map: Shared::new(HashMap::new().into()),
        }
    }
    pub fn map(&self) -> Shared<HashMap<String, Arc<dyn Attribute>>> {
        self.map.clone()
    }
    pub fn is_empty(&self) -> bool {
        self.map.rd().is_empty()
    }
    pub fn insert(&self, name: &str, attribute: Arc<dyn Attribute>) {
        self.map.wr().insert(name.to_string(), attribute);
    }
    pub fn get(&self, name: &str) -> Option<Arc<dyn Attribute>> {
        self.map.rd().get(name).cloned()
    }
    pub fn deep_clone(&self) -> Self {
        let mut out = HashMap::new();
        for (name, attribute) in self.map.rd().iter() {
            out.insert(name.to_string(), attribute.clone());
        }
        let map = Shared::new(out.into());
        Self { map }
    }
}

impl Display for Attributes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let map = self.map.rd().clone().into_iter();
        if map.len() != 0 {
            write!(f, "{{")?;
            for (i, attr) in map.enumerate() {
                if 0 < i {
                    write!(f, " ")?;
                }
                let (name, attribute) = attr;
                write!(f, "{name} = {attribute}")?;
            }
            write!(f, "}}")?;
        }
        Ok(())
    }
}

impl<T: ParserDispatch> Parser<T> {
    /// Parse a integer constant (e.g., `42 : i64`).
    pub fn parse_integer(&mut self) -> Result<IntegerAttr> {
        let integer = self.expect(TokenKind::Integer)?;
        let value = integer.lexeme;

        let _colon = self.expect(TokenKind::Colon)?;

        let num_bits = self.expect(TokenKind::IntType)?;
        let typ = IntegerType::from_str(&num_bits.lexeme).unwrap();
        let value = APInt::from_str(&num_bits.lexeme, &value);
        let integer = IntegerAttr::new(typ, value);
        Ok(integer)
    }
    /// Parse a string constant (e.g., `"hello"`).
    pub fn parse_string(&mut self) -> Result<StringAttr> {
        let string = self.expect(TokenKind::String)?;
        let text = string.lexeme.as_str();
        let text = text.trim_matches('"');
        Ok(StringAttr::from_str(text))
    }
    pub fn parse_attribute(&mut self) -> Result<Arc<dyn Attribute>> {
        let attribute = self.advance();
        let src = attribute.lexeme.clone();
        Ok(Arc::new(AnyAttr::from_str(&src)))
    }
    pub fn parse_attributes(&mut self) -> Result<Attributes> {
        let attributes = Attributes::new();
        self.expect(TokenKind::LBrace)?;
        while !self.check(TokenKind::RBrace) {
            let name = self.expect(TokenKind::BareIdentifier)?;
            let name = name.lexeme.clone();
            self.expect(TokenKind::Equal)?;
            let attribute = self.parse_attribute()?;
            attributes.insert(&name, attribute);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(attributes)
    }
}
