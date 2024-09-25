#![allow(dead_code)]

mod canonicalize;
mod dialect;
mod ir;
mod opt;
pub mod parser;
mod typ;

pub use canonicalize::canonicalize;
pub use ir::attribute::Attribute;
pub use ir::attribute::Attributes;
pub use ir::operation::Operation;
pub use ir::Block;
pub use opt::opt;
pub use parser::Parse;
pub use parser::Parser;

/// Dialects can define new operations, attributes, and types.
/// Each dialect is given an unique namespace that is prefixed.
///
/// Dialects can co-exist and can be produced and consumed by different passes.
pub trait Dialect {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
}
