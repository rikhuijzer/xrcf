use crate::parser::Parser;
use crate::typ::APInt;
use crate::typ::IntegerType;
use crate::Parse;
use std::fmt::Display;

/// Attributes belong to operations and can be used to, for example, specify
/// a SSA value.
pub trait Attribute {
    fn new(name: &'static str, value: &'static str) -> Self
    where
        Self: Sized;
    fn parse<T: Parse>(parser: &mut Parser<T>) -> Option<Self>
    where
        Self: Sized;

    fn name(&self) -> String;
    fn value(&self) -> &'static str;
    fn print(&self) -> String {
        format!("{} = {}", self.name(), self.value())
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for attribute in &self.attrs {
            write!(f, "{}", attribute.print())?;
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

impl Attribute for IntegerAttr {
    fn new(name: &'static str, value: &'static str) -> Self {
        todo!()
    }
    fn parse<T: Parse>(parser: &mut Parser<T>) -> Option<Self> {
        todo!()
    }
    fn name(&self) -> String {
        self.name.clone()
    }
    fn value(&self) -> &'static str {
        todo!()
    }
}
