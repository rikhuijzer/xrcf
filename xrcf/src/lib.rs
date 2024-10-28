mod canonicalize;
pub mod convert;
pub mod dialect;
pub mod ir;
pub mod parser;
pub mod targ3t;
mod transform;

pub use transform::default_passes;
pub use transform::init_subscriber;
pub use transform::transform;
pub use transform::DefaultTransformDispatch;
pub use transform::Passes;
pub use transform::TransformDispatch;

/// Dialects can define new operations, attributes, and types.
/// Each dialect is given an unique namespace that is prefixed.
///
/// Dialects can co-exist and can be produced and consumed by different passes.
pub trait Dialect {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
}
