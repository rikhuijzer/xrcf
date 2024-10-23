//! Intermediate representation (IR) for the compiler.
//!
//! These data structures are used as the basis for the compiler.
//! For example, this module some core types such as [Operation] and [Op].

mod attribute;
mod block;
mod module;
mod op;
mod op_operand;
mod operation;
mod region;
mod typ;
mod value;

pub use attribute::AnyAttr;
pub use attribute::Attribute;
pub use attribute::Attributes;
pub use attribute::IntegerAttr;
pub use attribute::StrAttr;
pub use block::Block;
pub use module::ModuleOp;
pub use op::Op;
pub use op_operand::OpOperand;
pub use op_operand::OpOperands;
pub use operation::display_region_inside_func;
pub use operation::Operation;
pub use operation::OperationName;
pub use region::Region;
pub use typ::APInt;
pub use typ::IntegerType;
pub use typ::PlaceholderType;
pub use typ::Type;
pub use typ::TypeConvert;
pub use typ::TypeParse;
pub use typ::Types;
pub use value::BlockArgument;
pub use value::OpResult;
pub use value::Users;
pub use value::Value;
pub use value::Values;

pub fn spaces(indent: i32) -> String {
    "  ".repeat(indent as usize)
}
