use crate::typ::IntegerType;
use crate::typ::APInt;

/// Attributes belong to operations and can be used to, for example, specify
/// a SSA value.
pub trait Attribute {
    fn new(name: &'static str, value: &'static str) -> Self
    where
        Self: Sized;
    fn parse(input: &str) -> Self
    where
        Self: Sized;

    fn name(&self) -> &'static str;
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

/// An attribute containing an integer value.
pub struct IntegerAttr {
    // The type of the integer like the precision.
    typ: IntegerType,
    // An arbitrary precision integer value.
    value: APInt,
}
