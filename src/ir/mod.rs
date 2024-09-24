pub mod attribute;
pub mod block;
pub mod module;
pub mod op;
pub mod operation;
pub mod region;
pub mod value;

pub use crate::ir::attribute::AnyAttr;
pub use crate::ir::attribute::StrAttr;
pub use crate::ir::block::Block;
pub use crate::ir::module::ModuleOp;
pub use crate::ir::op::Op;
pub use crate::ir::operation::Operation;
pub use crate::ir::operation::OperationName;
pub use crate::ir::region::Region;
pub use crate::ir::value::BlockArgument;
pub use crate::ir::value::OpResult;
pub use crate::ir::value::Type;
pub use crate::ir::value::Value;

pub fn spaces(indent: i32) -> String {
    "  ".repeat(indent as usize)
}
