#![allow(dead_code)]

mod canonicalize;
pub mod convert;
pub mod dialect;
pub mod ir;
mod opt;
pub mod parser;
pub mod targ3t;
mod typ;

pub use convert::Pass;
pub use ir::attribute::Attribute;
pub use ir::attribute::Attributes;
pub use ir::operation::Operation;
pub use ir::Block;
pub use opt::opt;
pub use opt::OptOptions;
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
