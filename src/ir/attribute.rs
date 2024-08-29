use crate::parser::Parser;
use crate::typ::APInt;
use crate::typ::IntegerType;
use crate::Parse;
use std::fmt::Display;

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

    fn name(&self) -> String;
    fn value(&self) -> &'static str;
    fn display(&self) -> String;
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for attribute in &self.attrs {
            write!(f, "{}", attribute.display())?;
        }
        Ok(())
    }
}

/// An attribute containing an integer value.
pub struct IntegerAttr {
    name: String,
    // The type of the integer like the precision.
    typ: IntegerType,
    // An arbitrary precision integer value.
    value: APInt,
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
    fn name(&self) -> String {
        self.name.clone()
    }
    fn value(&self) -> &'static str {
        todo!()
    }
    fn display(&self) -> String {
        self.value.clone()
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
    fn name(&self) -> String {
        self.name.clone()
    }
    fn value(&self) -> &'static str {
        todo!()
    }
    fn display(&self) -> String {
        self.value.clone()
    }
}
