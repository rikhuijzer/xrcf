#![allow(dead_code)]

mod arith;
mod op;
mod attribute;
mod typ;
mod translate;

pub use attribute::Attribute;
pub use attribute::Attributes;
pub use translate::translate;
pub use op::Op;

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
