use crate::parser::Parser;
use crate::typ::APInt;
use crate::typ::IntegerType;
use crate::Parse;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result;

/// Attributes are known-constant values of operations (a variable is not allowed).
/// Attributes belong to operations and can be used to, for example, specify
/// a SSA value.
pub trait Attribute {
    fn new(name: &str, value: &str) -> Self
    where
        Self: Sized;
    fn parse<T: Parse>(parser: &mut Parser<T>, name: &str) -> Option<Self>
    where
        Self: Sized;

    fn value(&self) -> String;
    fn display(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.value())
    }
}

impl Display for dyn Attribute {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.display(f)
    }
}

pub struct Attributes {
    attrs: Vec<Box<dyn Attribute>>,
}

impl Attributes {
    fn new(attrs: Vec<Box<dyn Attribute>>) -> Self {
        Self { attrs }
    }
}

impl Display for Attributes {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for attribute in &self.attrs {
            write!(f, "{}", attribute)?;
        }
        Ok(())
    }
}

/// An attribute containing an integer value.
pub struct IntegerAttr {
    // The type of the integer like the precision.
    typ: IntegerType,
    // An arbitrary precision integer value.
    value: APInt,
}

impl IntegerAttr {
    pub fn new(typ: IntegerType, value: APInt) -> Self {
        Self { typ, value }
    }
}

impl Attribute for IntegerAttr {
    fn new(_name: &str, value: &str) -> Self {
        Self {
            typ: IntegerType::new(64),
            value: APInt::new(64, value.parse::<u64>().unwrap(), true),
        }
    }
    fn parse<T: Parse>(_parser: &mut Parser<T>, _name: &str) -> Option<Self> {
        todo!()
    }
    fn value(&self) -> String {
        todo!()
    }
    fn display(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{} : {}", self.value, self.typ)
    }
}

pub struct StrAttr {
    name: String,
    value: String,
}

impl Attribute for StrAttr {
    fn new(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
        }
    }
    fn parse<T: Parse>(parser: &mut Parser<T>, name: &str) -> Option<Self> {
        let value = parser.advance();
        Some(Self {
            name: name.to_string(),
            value: value.lexeme.to_string(),
        })
    }
    fn value(&self) -> String {
        self.value.clone()
    }
    fn display(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.value)
    }
}

impl StrAttr {
    fn symbol_name(&self) -> String {
        self.value.clone()
    }
}

pub struct AnyAttr {
    name: String,
    value: String,
}

impl Attribute for AnyAttr {
    fn new(name: &str, value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
        }
    }
    fn parse<T: Parse>(parser: &mut Parser<T>, name: &str) -> Option<Self> {
        let value = parser.advance();
        Some(Self {
            name: name.to_string(),
            value: value.lexeme.to_string(),
        })
    }
    fn value(&self) -> String {
        self.value.clone()
    }
    fn display(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.value)
    }
}
