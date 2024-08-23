#![allow(dead_code)]

// Dialects can define new operations, attributes, and types.
// Each dialect is given an unique namespace that is prefixed.
//
// Dialects can co-exist and can be produced and consumed by different passes.
pub trait Dialect {
    fn new(name: &'static str, description: &'static str) -> Self;

    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
}

pub trait Attribute {
    fn new(name: &'static str, value: &'static str) -> Self;
    fn parse(input: &str) -> Self;

    fn name(&self) -> &'static str;
    fn value(&self) -> &'static str;
    fn print(&self) -> String {
        format!("{} = {}", self.name(), self.value())
    }
}

pub trait Attributes {
    fn new<A>(attrs: Vec<impl Attribute>) -> Self;
    fn attrs(&self) -> Vec<impl Attribute>;
    fn print(&self) -> String {
        self.attrs().iter().map(|attr| attr.print()).collect::<Vec<_>>().join(", ")
    }
}

// Takes attributes
pub trait Operation {
    // boxed attrs
    fn new(name: &'static str, attrs: impl Attributes) -> Self;

    fn name(&self) -> &'static str;
    fn print(&self) -> String {
        self.name().to_string()
    }
}

fn main() {
    println!("Hello, world!");
}
