#![allow(dead_code)]
#![allow(unused_variables)]

mod arith;
mod attribute;
mod typ;
mod compile;
mod ir;
mod parser;
mod dialect;

pub use attribute::Attribute;
pub use attribute::Attributes;
pub use compile::compile;
pub use ir::operation::Operation;

/// Dialects can define new operations, attributes, and types.
/// Each dialect is given an unique namespace that is prefixed.
///
/// Dialects can co-exist and can be produced and consumed by different passes.
pub trait Dialect {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
}

fn main() {
    println!("Hello, world!");
}
